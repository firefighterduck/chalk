use crate::search_graph::SearchGraph;
use crate::stack::{Stack, StackDepth};
use crate::{combine, Minimums, UCanonicalGoal};
use crate::{search_graph::DepthFirstNumber, ExFallible, ExSolution};
use crate::{
    solve::{SolveDatabase, SolveIteration},
    MaturityExtension,
};
use chalk_ir::interner::Interner;
use chalk_ir::Fallible;
use chalk_ir::{Canonical, ConstrainedSubst, Constraints, Goal, InEnvironment, UCanonical};
use chalk_solve::{coinductive_goal::IsCoinductive, RustIrDatabase, Solution};
use rustc_hash::FxHashMap;
use std::fmt;
use tracing::debug;
use tracing::{info, instrument};

type CacheWithStart<I> = (
    DepthFirstNumber,
    FxHashMap<UCanonicalGoal<I>, ExFallible<ExSolution<I>>>,
);

struct RecursiveContext<I: Interner> {
    stack: Stack,

    /// The "search graph" stores "in-progress results" that are still being
    /// solved.
    search_graph: SearchGraph<I>,

    /// The "cache" stores results for goals that we have completely solved.
    /// Things are added to the cache when we have completely processed their
    /// result.
    cache: FxHashMap<UCanonicalGoal<I>, Fallible<Solution<I>>>,

    /// Another cache for coinductive solutions that might be disproven later on.
    /// The DFN marks the start goal of the cycle.
    coinductive_cache: Option<CacheWithStart<I>>,

    /// The maximum size for goals.
    max_size: usize,

    caching_enabled: bool,
}

/// A Solver is the basic context in which you can propose goals for a given
/// program. **All questions posed to the solver are in canonical, closed form,
/// so that each question is answered with effectively a "clean slate"**. This
/// allows for better caching, and simplifies management of the inference
/// context.
struct Solver<'me, I: Interner> {
    program: &'me dyn RustIrDatabase<I>,
    context: &'me mut RecursiveContext<I>,
}

pub struct RecursiveSolver<I: Interner> {
    ctx: Box<RecursiveContext<I>>,
}

impl<I: Interner> RecursiveSolver<I> {
    pub fn new(overflow_depth: usize, max_size: usize, caching_enabled: bool) -> Self {
        Self {
            ctx: Box::new(RecursiveContext::new(
                overflow_depth,
                max_size,
                caching_enabled,
            )),
        }
    }
}

impl<I: Interner> fmt::Debug for RecursiveSolver<I> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "RecursiveSolver")
    }
}

/// An extension trait for merging `Result`s
trait MergeWith<T> {
    fn merge_with<F>(self, other: Self, f: F) -> Self
    where
        F: FnOnce(T, T) -> T;
}

impl<T> MergeWith<T> for ExFallible<T> {
    fn merge_with<F>(self: ExFallible<T>, other: ExFallible<T>, f: F) -> ExFallible<T>
    where
        F: FnOnce(T, T) -> T,
    {
        match (self, other) {
            (Err(_), Ok(v)) | (Ok(v), Err(_)) => Ok(v),
            (Ok(v1), Ok(v2)) => Ok(f(v1, v2)),
            (Err(_), Err(e)) => Err(e),
        }
    }
}

impl<I: Interner> RecursiveContext<I> {
    pub fn new(overflow_depth: usize, max_size: usize, caching_enabled: bool) -> Self {
        RecursiveContext {
            stack: Stack::new(overflow_depth),
            search_graph: SearchGraph::new(),
            cache: FxHashMap::default(),
            coinductive_cache: None,
            max_size,
            caching_enabled,
        }
    }

    pub(crate) fn solver<'me>(
        &'me mut self,
        program: &'me dyn RustIrDatabase<I>,
    ) -> Solver<'me, I> {
        Solver {
            program,
            context: self,
        }
    }
}

impl<'me, I: Interner> Solver<'me, I> {
    /// Solves a canonical goal. The substitution returned in the
    /// solution will be for the fully decomposed goal. For example, given the
    /// program
    ///
    /// ```ignore
    /// struct u8 { }
    /// struct SomeType<T> { }
    /// trait Foo<T> { }
    /// impl<U> Foo<u8> for SomeType<U> { }
    /// ```
    ///
    /// and the goal `exists<V> { forall<U> { SomeType<U>: Foo<V> }
    /// }`, `into_peeled_goal` can be used to create a canonical goal
    /// `SomeType<!1>: Foo<?0>`. This function will then return a
    /// solution with the substitution `?0 := u8`.
    pub(crate) fn solve_root_goal(
        &mut self,
        canonical_goal: &UCanonicalGoal<I>,
    ) -> Fallible<Solution<I>> {
        debug!("solve_root_goal(canonical_goal={:?})", canonical_goal);
        assert!(self.context.stack.is_empty());
        let minimums = &mut Minimums::new();
        self.solve_goal(canonical_goal.clone(), minimums).cut()
    }

    #[instrument(level = "debug", skip(self))]
    fn solve_new_subgoal(
        &mut self,
        canonical_goal: UCanonicalGoal<I>,
        depth: StackDepth,
        dfn: DepthFirstNumber,
    ) -> Minimums {
        // We start with `answer = None` and try to solve the goal. At the end of the iteration,
        // `answer` will be updated with the result of the solving process. If we detect a cycle
        // during the solving process, we cache `answer` and try to solve the goal again. We repeat
        // until we reach a fixed point for `answer`.
        // Considering the partial order:
        // - None < Some(Unique) < Some(Ambiguous)
        // - None < Some(CannotProve)
        // the function which maps the loop iteration to `answer` is a nondecreasing function
        // so this function will eventually be constant and the loop terminates.
        loop {
            let minimums = &mut Minimums::new();
            let (current_answer, current_prio) = self.solve_iteration(&canonical_goal, minimums);

            debug!(
                "solve_new_subgoal: loop iteration result = {:?} with minimums {:?}",
                current_answer, minimums
            );

            if !self.context.stack[depth].read_and_reset_cycle_flag() {
                // None of our subgoals depended on us directly.
                // We can return.
                self.context.search_graph[dfn].solution = current_answer;
                self.context.search_graph[dfn].solution_priority = current_prio;
                return *minimums;
            }

            let old_answer = &self.context.search_graph[dfn].solution;
            let old_prio = self.context.search_graph[dfn].solution_priority;

            // If the answer was at one point premature,
            // the final answer is also premature.
            let answer_maturity = current_answer.is_mature() && old_answer.is_mature();

            let (current_answer, current_prio) = combine::with_priorities_for_goal(
                self.program.interner(),
                &canonical_goal.canonical.value.goal,
                old_answer.clone().cut(),
                old_prio,
                current_answer.cut(),
                current_prio,
            );
            let current_answer_ex = ExFallible::extend(current_answer, Some(answer_maturity));

            // Some of our subgoals depended on us. We need to re-run
            // with the current answer.
            if self.context.search_graph[dfn].solution == current_answer_ex {
                // Reached a fixed point.
                return *minimums;
            }

            let current_answer_is_ambig = match &current_answer_ex {
                Ok(s) => s.is_ambig(),
                Err(_) => false,
            };

            self.context.search_graph[dfn].solution = current_answer_ex;
            self.context.search_graph[dfn].solution_priority = current_prio;

            // Subtle: if our current answer is ambiguous, we can just stop, and
            // in fact we *must* -- otherwise, we sometimes fail to reach a
            // fixed point. See `multiple_ambiguous_cycles` for more.
            if current_answer_is_ambig {
                return *minimums;
            }

            // Otherwise: rollback the search tree and try again.
            self.context.search_graph.rollback_to(dfn + 1);
        }
    }
}

impl<'me, I: Interner> SolveDatabase<I> for Solver<'me, I> {
    /// Attempt to solve a goal that has been fully broken down into leaf form
    /// and canonicalized. This is where the action really happens, and is the
    /// place where we would perform caching in rustc (and may eventually do in Chalk).
    #[instrument(level = "info", skip(self, minimums))]
    fn solve_goal(
        &mut self,
        goal: UCanonicalGoal<I>,
        minimums: &mut Minimums,
    ) -> ExFallible<ExSolution<I>> {
        // First check the cache.
        if let Some(value) = self.context.cache.get(&goal) {
            debug!("solve_reduced_goal: cache hit, value={:?}", value);
            return ExFallible::extend(value.clone(), None);
        }

        // Then check the cache for results dependent on a coinductive cycle.
        if let Some((_, ref mut cache)) = self.context.coinductive_cache {
            if let Some(value) = cache.get(&goal) {
                debug!(
                    "solve_reduced_goal: coinductive cache hit, value={:?}",
                    value
                );
                return value.clone();
            }
        }

        // Next, check if the goal is in the search tree already.
        if let Some(dfn) = self.context.search_graph.lookup(&goal) {
            // Check if this table is still on the stack.
            if let Some(depth) = self.context.search_graph[dfn].stack_depth {
                // Is this a coinductive goal? If so, that is success,
                // so we can return and set the minimum to its DFN.
                // Note that this return is not tabled. And so are
                // all other solutions in the cycle until the cycle
                // start is finished. This avoids prematurely cached
                // false positives.
                if self.context.stack.coinductive_cycle_from(depth) {
                    let value = ConstrainedSubst {
                        subst: goal.trivial_substitution(self.program.interner()),
                        constraints: Constraints::empty(self.program.interner()),
                    };
                    let solution = ExFallible::extend(
                        Ok(Solution::Unique(Canonical {
                            value,
                            binders: goal.canonical.binders.clone(),
                        })),
                        Some(false),
                    );

                    debug!("applying coinductive semantics");
                    debug!("assume solution {:?} for goal {:#?}", solution, goal);

                    if self.context.coinductive_cache.is_none() && self.context.caching_enabled {
                        self.context.coinductive_cache = Some((dfn, FxHashMap::default()));
                    }

                    if let Some((_, ref mut cache)) = self.context.coinductive_cache {
                        cache.insert(goal, solution.clone());
                    }

                    return solution;
                }

                self.context.stack[depth].flag_cycle();
            }

            minimums.update_from(self.context.search_graph[dfn].links);

            // Return the solution from the table.
            let previous_solution = self.context.search_graph[dfn].solution.clone();
            let previous_solution_priority = self.context.search_graph[dfn].solution_priority;
            info!(
                "solve_goal: cycle detected, previous solution {:?} with prio {:?}",
                previous_solution, previous_solution_priority
            );
            previous_solution
        } else {
            // Otherwise, push the goal onto the stack and create a table.
            // The initial result for this table is error.
            let coinductive_goal = goal.is_coinductive(self.program);
            let depth = self.context.stack.push(coinductive_goal);
            let dfn = self.context.search_graph.insert(&goal, depth);
            let subgoal_minimums = self.solve_new_subgoal(goal, depth, dfn);
            self.context.search_graph[dfn].links = subgoal_minimums;
            self.context.search_graph[dfn].stack_depth = None;
            self.context.stack.pop(depth);
            minimums.update_from(subgoal_minimums);

            // Read final result from table.
            let result = self.context.search_graph[dfn].solution.clone();
            let priority = self.context.search_graph[dfn].solution_priority;

            let mut coinductive_cache_cleared = false;

            // If processing this subgoal did not involve anything
            // outside of its subtree, then we can promote it to the
            // cache now. This is a sort of hack to alleviate the
            // worst of the repeated work that we do during tabling.
            if subgoal_minimums.positive >= dfn {
                if self.context.caching_enabled {
                    if let Some((start_dfn, ref mut coinductive_cache)) =
                        self.context.coinductive_cache
                    {
                        if dfn == start_dfn {
                            debug!("solve_reduced_goal: Coinductive cycle start encountered, moving temporary cache");
                            // If the start of a coinductive cycle is going to be cached (and the initial assumption holds),
                            // the additional cache can be transferred to the actual cache.
                            // Otherwise, if the assumption was disproved, the coinductive cache can be cleared.
                            if self.context.search_graph[dfn].solution.is_ok() {
                                for (goal, solution) in coinductive_cache.drain() {
                                    // Move all cached coinductive results that are not the assumption to the actual cache.
                                    if self
                                        .context
                                        .search_graph
                                        .lookup(&goal)
                                        .map(|dfn| dfn != start_dfn)
                                        .unwrap_or(true)
                                    {
                                        self.context.cache.insert(goal, solution.cut());
                                    }
                                }
                            } else {
                                coinductive_cache.clear();
                            }
                            coinductive_cache_cleared = true;

                            // Cache the actual result for the start of the coinductive cycle
                            self.context
                                .search_graph
                                .move_to_cache(dfn, &mut self.context.cache);
                        } else if !self.context.search_graph[dfn].solution.is_mature() {
                            self.context
                                .search_graph
                                .move_to_cache_ex(dfn, coinductive_cache);
                        } else {
                            self.context
                                .search_graph
                                .move_to_cache(dfn, &mut self.context.cache);
                        }
                    } else {
                        self.context
                            .search_graph
                            .move_to_cache(dfn, &mut self.context.cache);
                    }

                    debug!("solve_reduced_goal: SCC head encountered, moving to cache");
                } else {
                    debug!(
                        "solve_reduced_goal: SCC head encountered, rolling back as caching disabled"
                    );
                    self.context.search_graph.rollback_to(dfn);
                }
            }

            if coinductive_cache_cleared {
                self.context.coinductive_cache = None;
            }

            info!("solve_goal: solution = {:?} prio {:?}", result, priority);
            result
        }
    }

    fn interner(&self) -> &I {
        &self.program.interner()
    }

    fn db(&self) -> &dyn RustIrDatabase<I> {
        self.program
    }

    fn max_size(&self) -> usize {
        self.context.max_size
    }
}

impl<I: Interner> chalk_solve::Solver<I> for RecursiveSolver<I> {
    fn solve(
        &mut self,
        program: &dyn RustIrDatabase<I>,
        goal: &UCanonical<InEnvironment<Goal<I>>>,
    ) -> Option<chalk_solve::Solution<I>> {
        self.ctx.solver(program).solve_root_goal(goal).ok()
    }

    fn solve_limited(
        &mut self,
        program: &dyn RustIrDatabase<I>,
        goal: &UCanonical<InEnvironment<Goal<I>>>,
        _should_continue: &dyn std::ops::Fn() -> bool,
    ) -> Option<chalk_solve::Solution<I>> {
        // TODO support should_continue in recursive solver
        self.ctx.solver(program).solve_root_goal(goal).ok()
    }

    fn solve_multiple(
        &mut self,
        _program: &dyn RustIrDatabase<I>,
        _goal: &UCanonical<InEnvironment<Goal<I>>>,
        _f: &mut dyn FnMut(
            chalk_solve::SubstitutionResult<Canonical<ConstrainedSubst<I>>>,
            bool,
        ) -> bool,
    ) -> bool {
        unimplemented!("Recursive solver doesn't support multiple answers")
    }
}
