use std::ops::Deref;

use crate::search_graph::DepthFirstNumber;
use chalk_derive::HasInterner;
use chalk_ir::{interner::Interner, NoSolution};
use chalk_ir::{Goal, InEnvironment, UCanonical};
use chalk_solve::Solution;

pub type UCanonicalGoal<I> = UCanonical<InEnvironment<Goal<I>>>;

mod combine;
mod fulfill;
mod recursive;
mod search_graph;
pub mod solve;
mod stack;

pub use recursive::RecursiveSolver;

pub(crate) trait MaturityExtension {
    type Base;

    fn is_mature(&self) -> bool;

    fn cut(self) -> Self::Base;

    // None should lead to a mature result
    fn extend(b: Self::Base, mature: Option<bool>) -> Self;
}

/// Extended solution type with fitting NoSolution and Fallible to distinguish
/// between complete and premature results (i.e. depending on an assumption).
#[derive(Clone, Debug, PartialEq, Eq, HasInterner)]
pub(crate) enum ExSolution<I: Interner> {
    Mature(Solution<I>),
    Premature(Solution<I>),
}

impl<I: Interner> From<ExSolution<I>> for Solution<I> {
    fn from(ex_solution: ExSolution<I>) -> Self {
        match ex_solution {
            ExSolution::Mature(solution) => solution,
            ExSolution::Premature(solution) => solution,
        }
    }
}

impl<I: Interner> MaturityExtension for ExSolution<I> {
    type Base = Solution<I>;

    fn is_mature(&self) -> bool {
        matches!(self, Self::Mature(_))
    }

    fn cut(self) -> Self::Base {
        Solution::from(self)
    }

    fn extend(b: Self::Base, mature: Option<bool>) -> Self {
        if mature.unwrap_or(true) {
            Self::Mature(b)
        } else {
            Self::Premature(b)
        }
    }
}

impl<I: Interner> Deref for ExSolution<I> {
    type Target = Solution<I>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mature(solution) => solution,
            Self::Premature(solution) => solution,
        }
    }
}

impl<I: Interner> PartialEq<Solution<I>> for ExSolution<I> {
    fn eq(&self, other: &Solution<I>) -> bool {
        self.deref() == other
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ExNoSolution {
    NoMature,
    NoPremature,
}

impl MaturityExtension for ExNoSolution {
    type Base = NoSolution;

    fn is_mature(&self) -> bool {
        matches!(self, Self::NoMature)
    }

    fn cut(self) -> Self::Base {
        NoSolution
    }

    fn extend(_: Self::Base, mature: Option<bool>) -> Self {
        if mature.unwrap_or(true) {
            Self::NoMature
        } else {
            Self::NoPremature
        }
    }
}

pub(crate) type ExFallible<T> = Result<T, ExNoSolution>;

impl<T, E> MaturityExtension for Result<T, E>
where
    T: MaturityExtension,
    E: MaturityExtension,
{
    type Base = Result<T::Base, E::Base>;

    fn is_mature(&self) -> bool {
        match self {
            Ok(t) => t.is_mature(),
            Err(e) => e.is_mature(),
        }
    }

    fn cut(self) -> Self::Base {
        self.map(T::cut).map_err(E::cut)
    }

    fn extend(b: Self::Base, mature: Option<bool>) -> Self {
        b.map(|b| T::extend(b, mature))
            .map_err(|b| E::extend(b, mature))
    }
}

/// The `minimums` struct is used while solving to track whether we encountered
/// any cycles in the process.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Minimums {
    pub(crate) positive: DepthFirstNumber,
}

impl Minimums {
    pub fn new() -> Self {
        Minimums {
            positive: DepthFirstNumber::MAX,
        }
    }

    pub fn update_from(&mut self, minimums: Minimums) {
        self.positive = ::std::cmp::min(self.positive, minimums.positive);
    }
}
