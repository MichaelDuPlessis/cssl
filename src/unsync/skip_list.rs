use std::{borrow::Borrow, ops::Index, ptr::NonNull};

type Link<K, V> = Option<NonNull<Node<K, V>>>;

/// A `Node` in the underlying linked list.
#[derive(Debug)]
struct Node<K, V> {
    /// The key for the node.
    key: K,
    /// The value associated with a key.
    value: V,
    /// The next `Node`.
    next: Link<K, V>,
}

impl<K, V> Node<K, V> {
    /// Create a new `Node` with no next `Node` from a value.
    fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            next: None,
        }
    }

    /// Appends a `Node` to self
    fn append(&mut self, node: impl Into<Link<K, V>>) {
        self.next = node.into();
    }

    /// Get a refernce to the key
    fn key(&self) -> &K {
        &self.key
    }

    /// Get a refernce to the value
    fn value(&self) -> &V {
        &self.value
    }
}

impl<K, V> From<Node<K, V>> for Link<K, V> {
    fn from(value: Node<K, V>) -> Self {
        Some(unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) })
    }
}

/// The number of elements contained in each `Proxy`.
const PROXY_SIZE: usize = 4;

/// The P value for the `SkipList`, this is how many elements are skipped going up levels of the fast lane.
/// This will be known as the skipping factor
const P: usize = 4;

/// The number of fastlane levels in the `SkipList`.
const LEVELS: usize = 2;

/// Used as a Proxy between the fast lanes and the linked list
#[derive(Debug)]
struct Proxy<K, V> {
    // The values that map to links
    values: [K; PROXY_SIZE],
    // The links to nodes in the `LinkedList`
    links: [Link<K, V>; PROXY_SIZE],
}

/// A reference to a fast lane
#[derive(Debug)]
pub struct Lane<'a, K> {
    lane: &'a [K],
}

impl<'a, K> Lane<'a, K> {
    /// Create a new `Lane`.
    fn new(lane: &'a [K]) -> Self {
        Self { lane }
    }

    /// Get the length of the `Lane`.
    fn len(&self) -> usize {
        self.lane.len()
    }

    /// Find the index in the `Lane` that contains the passed in key or the index - 1 of the value just larger then it.
    /// It is assumed the `Lane` is sorted in ascending order.
    fn find<Q>(&self, key: &Q) -> usize
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        self.lane.iter().take_while(|&k| key <= k.borrow()).count()
    }

    // TODO: Maybe implement index and have a get_unchecked
    /// Get access to the inncer slice
    fn inner(&self) -> &[K] {
        self.lane
    }
}

/// Is all the fast lanes used in the `SkipListMap`.
#[derive(Debug, Default)]
pub struct Lanes<T> {
    lanes: Vec<T>,
}

impl<T> Lanes<T> {
    /// Create a new empty set of `Lanes`.
    pub fn new() -> Self {
        Self { lanes: Vec::new() }
    }
    /// Calculates the number of elements at a specific level.
    /// The formula is as follows ceil(Total Elements / ((Skipping Factor) ^ Level))
    fn lane_len(&self, level: usize) -> usize {
        // remember we start 0 indexed
        let level_power = P.pow(level as u32 + 1);
        // This is a cheap way to perform ceiling operations on integers
        (self.lanes.len() + level_power - 1) / level_power
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
    fn lane_start(&self, level: usize) -> usize {
        // The start of a level is the sum of the lengths of the previous levels
        // taking the formula for the level_len summing it an simplifying leads to an
        // equation with no loops, aint that cool
        // remember we start 0 indexed

        let pk = P.pow(level as u32);
        level + (self.lanes.len() - 1) * ((pk - 1) / P - 1)
    }

    /// Retrieve fast lane located on the nth level.
    fn lane(&self, level: usize) -> Lane<'_, T> {
        // if it is the highest level we start at
        // Calculating the beginning and end of the level
        // 1. calculate the number of elements in the level
        let num_elements = self.lane_len(level);

        // 2. find the start of the level
        let level_start = self.lane_start(level);

        Lane::new(&self.lanes[level_start..level_start + num_elements])
    }

    /// The number of fast lanes
    // TODO: Maybe change to a smaller size like u8 or u32
    fn num_lanes(&self) -> usize {
        LEVELS
    }
}

/// A Cache Sensitive Skip List
#[derive(Debug, Default)]
pub struct SkipListMap<K, V> {
    /// The fast lanes.
    lanes: Lanes<K>,
    /// The proxy list.
    proxy_list: Vec<Proxy<K, V>>,
    /// The linked list containing all the nodes.
    linked_list: Link<K, V>,
    /// The number of elements in the `SkipListMap`.
    len: usize,
}

impl<K, V> SkipListMap<K, V> {
    /// Create a new `SkipListMap`.
    pub fn new() -> Self {
        Self {
            lanes: Lanes::new(),
            proxy_list: Vec::new(),
            linked_list: None,
            len: 0,
        }
    }

    /// The number of elements in the `SkipListMap`.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<K, V> SkipListMap<K, V>
where
    K: Ord,
{
    /// Retrieves an item from the `SkipListMap`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let key = key.borrow();

        for level in LEVELS - 1..=0 {
            // get the lane that corresponding to that level
            let lane = self.lanes.lane(level);
            // get the index in the lane that is the key or just less than the key
            let index = lane.find(key);
            // check if the index contains the key
            if unsafe { lane.inner().get_unchecked(index) }.borrow() == key {}
        }

        todo!()
    }

    /// Insert a key value pair into the `SkipListMap`.
    /// Returns the value previously associated with the key and replaces it with the new one if a previous exists.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        todo!()
    }
}
