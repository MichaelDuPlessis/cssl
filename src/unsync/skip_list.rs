use crate::unsync::linked_list::{Link, LinkedList};

/// The number of elements contained in each `Proxy`.
const PROXY_SIZE: usize = 4;

/// The P value for the `SkipList`, this is how many elements are skipped going up levels of the fast lane.
/// This will be known as the skipping factor
const P: usize = 4;

/// The number of fastlane levels in the `SkipList`.
const LEVELS: usize = 2;

/// Used as a Proxy between the fast lanes and the linked list
#[derive(Debug)]
struct Proxy<T> {
    // The values that map to links
    values: [T; PROXY_SIZE],
    // The links to nodes in the `LinkedList`
    links: [Link<T>; PROXY_SIZE],
}

/// A Cache Sensitive Skip List
#[derive(Debug, Default)]
pub struct SkipList<T> {
    /// The fast lanes.
    lanes: Vec<T>,
    /// The proxy list.
    proxy_list: Vec<Proxy<T>>,
    /// The linked list containing all the nodes.
    linked_list: LinkedList<T>,
    /// The number of elements in the `SkipList`.
    len: usize,
}

impl<T> SkipList<T> {
    /// Create a new `SkipList`.
    pub fn new() -> Self {
        Self {
            lanes: Vec::new(),
            proxy_list: Vec::new(),
            linked_list: LinkedList::new(),
            len: 0,
        }
    }

    /// Calculates the number of elements at a specific level.
    /// The formula is as follows ceil(Total Elements / ((Skipping Factor) ^ Level))
    fn level_len(&self, level: usize) -> usize {
        // remember we start 0 indexed
        let level_power = P.pow(level as u32 + 1);
        // This is a cheap way to perform ceiling operations on integers
        (self.len() + level_power - 1) / level_power
    }

    /// Calculates the start index of a specific level
    /// Compute S(k) = sum_{i=1..k} (N + P^i - 1) / P^i
    ///
    /// Derivation:
    ///   (N + P^i - 1) / P^i
    /// = N / P^i + (P^i / P^i) - (1 / P^i)
    /// = 1 + (N - 1) / P^i
    ///
    /// So:
    ///   S(k) = sum_{i=1..k} [ 1 + (N - 1) / P^i ]
    ///        = k + (N - 1) * sum_{i=1..k} (1 / P^i)
    ///
    /// The inner sum is a finite geometric series:
    ///   sum_{i=1..k} 1/P^i = (1 - P^(-k)) / (P - 1)
    ///
    /// Multiply top and bottom by P^k to make it integer-friendly:
    ///   (1 - P^(-k)) / (P - 1)
    /// = (P^k - 1) / (P^k * (P - 1))
    ///
    /// Final formula:
    ///   S(k) = k + (N - 1) * (P^k - 1) / (P^k * (P - 1))
    ///
    /// Special case: if P = 1, then each term = N, so S(k) = k * N.
    fn level_start(&self, level: usize) -> usize {
        // The start of a level is the sum of the lengths of the previous levels
        // taking the formula for the level_len summing it an simplifying leads to an
        // equation with no loops, aint that cool
        // remember we start 0 indexed

        let pk = P.pow(level as u32);
        level + (self.len() - 1) * ((pk - 1) / P - 1)
    }

    /// Retrieve fast lane located on the nth level.
    fn level(&self, level: usize) -> &[T] {
        // if it is the highest level we start at
        // Calculating the beginning and end of the level
        // 1. calculate the number of elements in the level
        let num_elements = self.level_len(level);

        // 2. find the start of the level
        let level_start = self.level_start(level);

        &self.lanes[level_start..level_start + num_elements]
    }

    /// The number of elements in the `SkipList`.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> SkipList<T>
where
    T: PartialOrd + PartialEq,
{
    /// Insertes an element into the `SkipList`.
    pub fn insert(&mut self, elem: T) {}

    /// Finds the `Node` in the `SkipList` that is either just before the next greatest `Node` or equal to the value
    fn find(&self, elem: &T) -> Link<T> {
        for level in LEVELS - 1..=0 {}

        todo!()
    }
}
