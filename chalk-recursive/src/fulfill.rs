use crate::solve::SolveDatabase;
use crate::{ExFallible, ExNoSolution, ExSolution, MaturityExtension, Minimums};
use chalk_ir::fold::Fold;
use chalk_ir::interner::{HasInterner, Interner};
use chalk_ir::visit::Visit;
use chalk_ir::zip::Zip;
use chalk_ir::{cast::Cast, NoSolution};
use chalk_ir::{
    Binders, Canonical, ConstrainedSubst, Constraint, Constraints, DomainGoal, Environment, EqGoal,
    Fallible, GenericArg, Goal, GoalData, InEnvironment, ProgramClauseImplication, QuantifierKind,
    Substitution, SubtypeGoal, TyKind, TyVariableKind, UCanonical, UnificationDatabase,
    UniverseMap, Variance,
};
use chalk_solve::debug_span;
use chalk_solve::infer::{InferenceTable, ParameterEnaVariableExt};
use chalk_solve::solve::truncate;
use chalk_solve::{Guidance, Solution};
use rustc_hash::FxHashSet;
use std::fmt::Debug;
use tracing::{debug, instrument};

enum Outcome {
    Complete(bool),
    Incomplete(bool),
}

impl Outcome {
    fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }
}

impl MaturityExtension for Outcome {
    type Base = Outcome;

    fn is_mature(&self) -> bool {
        match self {
            &Self::Complete(mature) => mature,
            &Self::Incomplete(mature) => mature,
        }
    }

    fn cut(self) -> Self::Base {
        self
    }

    fn extend(b: Self::Base, mature: Option<bool>) -> Self {
        let mature = mature.unwrap_or(true);
        match b {
            Self::Complete(_) => Self::Complete(mature),
            Self::Incomplete(_) => Self::Incomplete(mature),
        }
    }
}

/// A goal that must be resolved
#[derive(Clone, Debug, PartialEq, Eq)]
enum Obligation<I: Interner> {
    /// For "positive" goals, we flatten all the way out to leafs within the
    /// current `Fulfill`
    Prove(InEnvironment<Goal<I>>),

    /// For "negative" goals, we don't flatten in *this* `Fulfill`, which would
    /// require having a logical "or" operator. Instead, we recursively solve in
    /// a fresh `Fulfill`.
    Refute(InEnvironment<Goal<I>>),
}

/// When proving a leaf goal, we record the free variables that appear within it
/// so that we can update inference state accordingly.
#[derive(Clone, Debug)]
struct PositiveSolution<I: Interner> {
    free_vars: Vec<GenericArg<I>>,
    universes: UniverseMap,
    solution: ExSolution<I>,
}

/// When refuting a goal, there's no impact on inference state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NegativeSolution {
    Refuted(bool),
    Ambiguous(bool),
}

impl MaturityExtension for NegativeSolution {
    type Base = NegativeSolution;

    fn is_mature(&self) -> bool {
        match self {
            &Self::Refuted(mature) => mature,
            &Self::Ambiguous(mature) => mature,
        }
    }

    fn cut(self) -> Self::Base {
        self
    }

    fn extend(b: Self::Base, mature: Option<bool>) -> Self {
        let mature = mature.unwrap_or(true);
        match b {
            Self::Refuted(_) => Self::Refuted(mature),
            Self::Ambiguous(_) => Self::Ambiguous(mature),
        }
    }
}

fn canonicalize<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: &I,
    value: T,
) -> (Canonical<T::Result>, Vec<GenericArg<I>>)
where
    T: Fold<I>,
    T::Result: HasInterner<Interner = I>,
{
    let res = infer.canonicalize(interner, value);
    let free_vars = res
        .free_vars
        .into_iter()
        .map(|free_var| free_var.to_generic_arg(interner))
        .collect();
    (res.quantified, free_vars)
}

fn u_canonicalize<I: Interner, T>(
    _infer: &mut InferenceTable<I>,
    interner: &I,
    value0: &Canonical<T>,
) -> (UCanonical<T::Result>, UniverseMap)
where
    T: Clone + HasInterner<Interner = I> + Fold<I> + Visit<I>,
    T::Result: HasInterner<Interner = I>,
{
    let res = InferenceTable::u_canonicalize(interner, value0);
    (res.quantified, res.universes)
}

fn unify<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: &I,
    db: &dyn UnificationDatabase<I>,
    environment: &Environment<I>,
    variance: Variance,
    a: &T,
    b: &T,
) -> Fallible<Vec<InEnvironment<Goal<I>>>>
where
    T: ?Sized + Zip<I>,
{
    let res = infer.relate(interner, db, environment, variance, a, b)?;
    Ok(res.goals)
}

/// A `Fulfill` is where we actually break down complex goals, instantiate
/// variables, and perform inference. It's highly stateful. It's generally used
/// in Chalk to try to solve a goal, and then package up what was learned in a
/// stateless, canonical way.
///
/// In rustc, you can think of there being an outermost `Fulfill` that's used when
/// type checking each function body, etc. There, the state reflects the state
/// of type inference in general. But when solving trait constraints, *fresh*
/// `Fulfill` instances will be created to solve canonicalized, free-standing
/// goals, and transport what was learned back to the outer context.
pub(super) struct Fulfill<'s, I: Interner, Solver: SolveDatabase<I>> {
    solver: &'s mut Solver,
    subst: Substitution<I>,
    infer: InferenceTable<I>,

    /// The remaining goals to prove or refute
    obligations: Vec<Obligation<I>>,

    /// Lifetime constraints that must be fulfilled for a solution to be fully
    /// validated.
    constraints: FxHashSet<InEnvironment<Constraint<I>>>,

    /// Record that a goal has been processed that can neither be proved nor
    /// refuted. In such a case the solution will be either `CannotProve`, or `Err`
    /// in the case where some other goal leads to an error.
    cannot_prove: bool,
}

impl<'s, I: Interner, Solver: SolveDatabase<I>> Fulfill<'s, I, Solver> {
    #[instrument(level = "debug", skip(solver, infer))]
    pub(super) fn new_with_clause(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<DomainGoal<I>>,
        clause: &Binders<ProgramClauseImplication<I>>,
    ) -> ExFallible<Self> {
        let mut fulfill = Fulfill {
            solver,
            infer,
            subst,
            obligations: vec![],
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        let ProgramClauseImplication {
            consequence,
            conditions,
            constraints,
            priority: _,
        } = fulfill
            .infer
            .instantiate_binders_existentially(fulfill.solver.interner(), clause.clone());

        debug!(?consequence, ?conditions, ?constraints);
        fulfill
            .constraints
            .extend(constraints.as_slice(fulfill.interner()).to_owned());

        debug!("the subst is {:?}", fulfill.subst);

        fulfill.unify(
            &canonical_goal.environment,
            Variance::Invariant,
            &canonical_goal.goal,
            &consequence,
        )?;

        // if so, toss in all of its premises
        for condition in conditions.as_slice(fulfill.solver.interner()) {
            fulfill.push_goal(&canonical_goal.environment, condition.clone())?;
        }

        Ok(fulfill)
    }

    pub(super) fn new_with_simplification(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<Goal<I>>,
    ) -> ExFallible<Self> {
        let mut fulfill = Fulfill {
            solver,
            infer,
            subst,
            obligations: vec![],
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        fulfill.push_goal(&canonical_goal.environment, canonical_goal.goal.clone())?;

        Ok(fulfill)
    }

    fn push_obligation(&mut self, obligation: Obligation<I>) {
        // truncate to avoid overflows
        match &obligation {
            Obligation::Prove(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    return;
                }
            }
            Obligation::Refute(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    return;
                }
            }
        };
        self.obligations.push(obligation);
    }

    /// Unifies `a` and `b` in the given environment.
    ///
    /// Wraps `InferenceTable::unify`; any resulting normalizations are added
    /// into our list of pending obligations with the given environment.
    pub(super) fn unify<T>(
        &mut self,
        environment: &Environment<I>,
        variance: Variance,
        a: &T,
        b: &T,
    ) -> ExFallible<()>
    where
        T: ?Sized + Zip<I> + Debug,
    {
        let goals = unify(
            &mut self.infer,
            self.solver.interner(),
            self.solver.db().unification_database(),
            environment,
            variance,
            a,
            b,
        )
        .map_err(|e| ExNoSolution::extend(e, None))?;
        debug!("unify({:?}, {:?}) succeeded", a, b);
        debug!("unify: goals={:?}", goals);
        for goal in goals {
            let goal = goal.cast(self.solver.interner());
            self.push_obligation(Obligation::Prove(goal));
        }
        Ok(())
    }

    /// Create obligations for the given goal in the given environment. This may
    /// ultimately create any number of obligations.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn push_goal(
        &mut self,
        environment: &Environment<I>,
        goal: Goal<I>,
    ) -> ExFallible<()> {
        let interner = self.solver.interner();
        match goal.data(interner) {
            GoalData::Quantified(QuantifierKind::ForAll, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_universally(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Quantified(QuantifierKind::Exists, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_existentially(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Implies(wc, subgoal) => {
                let new_environment =
                    &environment.add_clauses(interner, wc.iter(interner).cloned());
                self.push_goal(new_environment, subgoal.clone())?;
            }
            GoalData::All(goals) => {
                for subgoal in goals.as_slice(interner) {
                    self.push_goal(environment, subgoal.clone())?;
                }
            }
            GoalData::Not(subgoal) => {
                let in_env = InEnvironment::new(environment, subgoal.clone());
                self.push_obligation(Obligation::Refute(in_env));
            }
            GoalData::DomainGoal(_) => {
                let in_env = InEnvironment::new(environment, goal);
                self.push_obligation(Obligation::Prove(in_env));
            }
            GoalData::EqGoal(EqGoal { a, b }) => {
                self.unify(&environment, Variance::Invariant, &a, &b)?;
            }
            GoalData::SubtypeGoal(SubtypeGoal { a, b }) => {
                let a_norm = self.infer.normalize_ty_shallow(interner, a);
                let a = a_norm.as_ref().unwrap_or(a);
                let b_norm = self.infer.normalize_ty_shallow(interner, b);
                let b = b_norm.as_ref().unwrap_or(b);

                if matches!(
                    a.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) && matches!(
                    b.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) {
                    self.cannot_prove = true;
                } else {
                    self.unify(&environment, Variance::Covariant, &a, &b)?;
                }
            }
            GoalData::CannotProve => {
                debug!("Pushed a CannotProve goal, setting cannot_prove = true");
                self.cannot_prove = true;
            }
        }
        Ok(())
    }

    #[instrument(level = "debug", skip(self, minimums))]
    fn prove(
        &mut self,
        wc: InEnvironment<Goal<I>>,
        minimums: &mut Minimums,
    ) -> ExFallible<PositiveSolution<I>> {
        let interner = self.solver.interner();
        let (quantified, free_vars) = canonicalize(&mut self.infer, interner, wc);
        let (quantified, universes) = u_canonicalize(&mut self.infer, interner, &quantified);
        let result = self.solver.solve_goal(quantified, minimums);
        Ok(PositiveSolution {
            free_vars,
            universes,
            solution: result?,
        })
    }

    fn refute(&mut self, goal: InEnvironment<Goal<I>>) -> ExFallible<NegativeSolution> {
        let canonicalized = match self
            .infer
            .invert_then_canonicalize(self.solver.interner(), goal)
        {
            Some(v) => v,
            None => {
                // Treat non-ground negatives as ambiguous. Note that, as inference
                // proceeds, we may wind up with more information here.
                return Ok(NegativeSolution::Ambiguous(false));
            }
        };

        // Negate the result
        let (quantified, _) =
            u_canonicalize(&mut self.infer, self.solver.interner(), &canonicalized);
        let mut minimums = Minimums::new(); // FIXME -- minimums here seems wrong
        let solution = self.solver.solve_goal(quantified, &mut minimums);
        if let Ok(solution) = solution {
            let solution_mature = solution.is_mature();
            if solution.is_unique() {
                Err(ExNoSolution::extend(NoSolution, Some(solution_mature)))
            } else {
                Ok(NegativeSolution::Ambiguous(solution_mature))
            }
        } else {
            Ok(NegativeSolution::Refuted(solution.is_mature()))
        }
    }

    /// Trying to prove some goal led to a the substitution `subst`; we
    /// wish to apply that substitution to our own inference variables
    /// (and incorporate any region constraints). This substitution
    /// requires some mapping to get it into our namespace -- first,
    /// the universes it refers to have been canonicalized, and
    /// `universes` stores the mapping back into our
    /// universes. Second, the free variables that appear within can
    /// be mapped into our variables with `free_vars`.
    fn apply_solution(
        &mut self,
        free_vars: Vec<GenericArg<I>>,
        universes: UniverseMap,
        subst: Canonical<ConstrainedSubst<I>>,
    ) {
        use chalk_solve::infer::ucanonicalize::UniverseMapExt;
        let subst = universes.map_from_canonical(self.interner(), &subst);
        let ConstrainedSubst { subst, constraints } = self
            .infer
            .instantiate_canonical(self.solver.interner(), subst);

        debug!(
            "fulfill::apply_solution: adding constraints {:?}",
            constraints
        );
        self.constraints
            .extend(constraints.as_slice(self.interner()).to_owned());

        // We use the empty environment for unification here because we're
        // really just doing a substitution on unconstrained variables, which is
        // guaranteed to succeed without generating any new constraints.
        let empty_env = &Environment::new(self.solver.interner());

        for (i, free_var) in free_vars.into_iter().enumerate() {
            let subst_value = subst.at(self.interner(), i);
            self.unify(empty_env, Variance::Invariant, &free_var, subst_value)
                .unwrap_or_else(|err| {
                    panic!(
                        "apply_solution failed with free_var={:?}, subst_value={:?}: {:?}",
                        free_var, subst_value, err
                    );
                });
        }
    }

    fn fulfill(&mut self, minimums: &mut Minimums) -> ExFallible<Outcome> {
        debug_span!("fulfill", obligations=?self.obligations);

        // Try to solve all the obligations. We do this via a fixed-point
        // iteration. We try to solve each obligation in turn. Anything which is
        // successful, we drop; anything ambiguous, we retain in the
        // `obligations` array. This process is repeated so long as we are
        // learning new things about our inference state.
        let mut obligations = Vec::with_capacity(self.obligations.len());
        let mut progress = true;
        let mut outcome_mature = true;

        while progress {
            progress = false;
            debug!("start of round, {} obligations", self.obligations.len());

            // Take the list of `obligations` to solve this round and replace it
            // with an empty vector. Iterate through each obligation to solve
            // and solve it if we can. If not (because of ambiguity), then push
            // it back onto `self.to_prove` for next round. Note that
            // `solve_one` may also push onto the `self.to_prove` list
            // directly.
            assert!(obligations.is_empty());
            while let Some(obligation) = self.obligations.pop() {
                let ambiguous = match &obligation {
                    Obligation::Prove(wc) => {
                        let PositiveSolution {
                            free_vars,
                            universes,
                            solution,
                        } = self.prove(wc.clone(), minimums)?;

                        if let Some(constrained_subst) = solution.definite_subst(self.interner()) {
                            // If the substitution is empty, we won't actually make any progress by applying it!
                            // So we need to check this to prevent endless loops.
                            // (An ambiguous solution with empty substitution
                            // can probably not happen in valid code, but it can
                            // happen e.g. when there are overlapping impls.)
                            if !constrained_subst.value.subst.is_empty(self.interner())
                                || !constrained_subst
                                    .value
                                    .constraints
                                    .is_empty(self.interner())
                            {
                                self.apply_solution(free_vars, universes, constrained_subst);
                                progress = true;
                            }
                        }

                        // If at least one of the subgoals returned an premature result
                        // then the whole outcome is premature.
                        outcome_mature &= solution.is_mature();

                        solution.is_ambig()
                    }
                    Obligation::Refute(goal) => {
                        let answer = self.refute(goal.clone())?;
                        outcome_mature &= answer.is_mature();
                        answer == NegativeSolution::Ambiguous(outcome_mature)
                    }
                };

                if ambiguous {
                    debug!("ambiguous result: {:?}", obligation);
                    obligations.push(obligation);
                }
            }

            self.obligations.extend(obligations.drain(..));
            debug!("end of round, {} obligations left", self.obligations.len());
        }

        // At the end of this process, `self.obligations` should have
        // all of the ambiguous obligations, and `obligations` should
        // be empty.
        assert!(obligations.is_empty());

        if self.obligations.is_empty() {
            Ok(Outcome::Complete(outcome_mature))
        } else {
            Ok(Outcome::Incomplete(outcome_mature))
        }
    }

    /// Try to fulfill all pending obligations and build the resulting
    /// solution. The returned solution will transform `subst` substitution with
    /// the outcome of type inference by updating the replacements it provides.
    pub(super) fn solve(mut self, minimums: &mut Minimums) -> ExFallible<ExSolution<I>> {
        let outcome = self.fulfill(minimums)?;

        if self.cannot_prove {
            debug!("Goal cannot be proven (cannot_prove = true), returning ambiguous");
            return Ok(ExSolution::extend(
                Solution::Ambig(Guidance::Unknown),
                Some(outcome.is_mature()),
            ));
        }

        if outcome.is_complete() {
            // No obligations remain, so we have definitively solved our goals,
            // and the current inference state is the unique way to solve them.

            let constraints = Constraints::from_iter(self.interner(), self.constraints.clone());
            let constrained = canonicalize(
                &mut self.infer,
                self.solver.interner(),
                ConstrainedSubst {
                    subst: self.subst,
                    constraints,
                },
            );
            return Ok(ExSolution::extend(
                Solution::Unique(constrained.0),
                Some(outcome.is_mature()),
            ));
        }

        // Otherwise, we have (positive or negative) obligations remaining, but
        // haven't proved that it's *impossible* to satisfy out obligations. we
        // need to determine how to package up what we learned about type
        // inference as an ambiguous solution.

        let canonical_subst =
            canonicalize(&mut self.infer, self.solver.interner(), self.subst.clone());

        if canonical_subst
            .0
            .value
            .is_identity_subst(self.solver.interner())
        {
            // In this case, we didn't learn *anything* definitively. So now, we
            // go one last time through the positive obligations, this time
            // applying even *tentative* inference suggestions, so that we can
            // yield these upwards as our own suggestions. There are no
            // particular guarantees about *which* obligaiton we derive
            // suggestions from.

            while let Some(obligation) = self.obligations.pop() {
                if let Obligation::Prove(goal) = obligation {
                    let PositiveSolution {
                        free_vars,
                        universes,
                        solution,
                    } = self.prove(goal, minimums).unwrap();
                    if let Some(constrained_subst) =
                        solution.constrained_subst(self.solver.interner())
                    {
                        self.apply_solution(free_vars, universes, constrained_subst);
                        return Ok(ExSolution::extend(
                            Solution::Ambig(Guidance::Suggested(canonical_subst.0)),
                            Some(solution.is_mature()),
                        ));
                    }
                }
            }

            Ok(ExSolution::extend(
                Solution::Ambig(Guidance::Unknown),
                Some(outcome.is_mature()),
            ))
        } else {
            // While we failed to prove the goal, we still learned that
            // something had to hold. Here's an example where this happens:
            //
            // ```rust
            // trait Display {}
            // trait Debug {}
            // struct Foo<T> {}
            // struct Bar {}
            // struct Baz {}
            //
            // impl Display for Bar {}
            // impl Display for Baz {}
            //
            // impl<T> Debug for Foo<T> where T: Display {}
            // ```
            //
            // If we pose the goal `exists<T> { T: Debug }`, we can't say
            // for sure what `T` must be (it could be either `Foo<Bar>` or
            // `Foo<Baz>`, but we *can* say for sure that it must be of the
            // form `Foo<?0>`.
            Ok(ExSolution::extend(
                Solution::Ambig(Guidance::Definite(canonical_subst.0)),
                Some(outcome.is_mature()),
            ))
        }
    }

    fn interner(&self) -> &I {
        self.solver.interner()
    }
}
