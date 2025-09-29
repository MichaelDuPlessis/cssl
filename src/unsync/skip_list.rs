use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, ptr::NonNull};

/// The number of elements contained in each `Proxy`.
const PROXY_SIZE: usize = 4;

/// The P value for the `SkipList`, this is how many elements are skipped going up levels of the fast lane.
/// This will be known as the skipping factor
const P: usize = 4;

/// The number of fastlane levels in the `SkipList`.
const LEVELS: usize = 2;

/// An key: value pair stored in the `SkipListMap`.
struct Item<K, V> {
    key: K,
    value: V,
}

impl<K, V> Item<K, V> {
    /// Create a new item from a key and value
    fn new(key: K, value: V) -> Self {
        Self { key, value }
    }

    /// Retrieve a reference to the key.
    fn key(&self) -> &K {
        &self.key
    }

    /// Retrieve a reference to the value.
    fn value(&self) -> &V {
        &self.value
    }

    /// Replaces the current value with the passed in value and returns the old value.
    fn replace(&mut self, value: V) -> V {
        std::mem::replace(&mut self.value, value)
    }
}

impl<K, V> Debug for Item<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Item")
            .field("key", &self.key)
            .field("value", &self.value)
            .finish()
    }
}

type Link<K, V> = NonNull<Node<K, V>>;

/// A `Node` in the underlying linked list.
#[derive(Debug)]
struct Node<K, V> {
    item: Item<K, V>,
    /// The next `Node`.
    next: Option<Link<K, V>>,
}

impl<K, V> Node<K, V> {
    /// Create a new `Node` with no next `Node` from a value.
    fn new(key: K, value: V) -> Self {
        Self {
            item: Item::new(key, value),
            next: None,
        }
    }

    /// Appends a `Node` to self.
    fn append(&mut self, node: impl Into<Option<Link<K, V>>>) {
        self.next = node.into();
    }

    /// Get a reference to the item.
    fn item(&self) -> &Item<K, V> {
        &self.item
    }

    /// Get a mutable reference to the item.
    fn item_mut(&mut self) -> &mut Item<K, V> {
        &mut self.item
    }

    /// Get a refernce to the key.
    fn key(&self) -> &K {
        self.item.key()
    }

    /// Get a refernce to the value.
    fn value(&self) -> &V {
        self.item.value()
    }

    /// Get a pointer to the next node if there is one
    fn next(&self) -> Option<Link<K, V>> {
        self.next
    }

    /// Create an iterator over immutable references
    fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            current: Some(self.into()),
            _marker: PhantomData,
        }
    }
}

impl<K, V> From<Node<K, V>> for Option<Link<K, V>> {
    fn from(value: Node<K, V>) -> Self {
        Some(unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) })
    }
}

/// Immutabale iterator for `LinkedList`.
struct Iter<'a, K: 'a, V: 'a> {
    current: Option<Link<K, V>>,
    _marker: PhantomData<&'a Link<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = &'a Item<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.current?; // copy out the link (Link<T> is Copy)
        let node = unsafe { link.as_ref() };

        // Advance the iterator
        self.current = node.next();

        Some(node.item())
    }
}

impl<'a, K, V> IntoIterator for &'a Node<K, V> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Used as a Proxy between the fast lanes and the linked list
#[derive(Debug)]
struct Proxy<K, V> {
    // The links to nodes in the `LinkedList`
    links: [Link<K, V>; PROXY_SIZE],
}

impl<K, V> Proxy<K, V> {
    /// Find the `Link` which is the passed in key.
    fn get<Q>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // There is a unchecked unwrapping but if we some how have a link that points to nothing we messed up somewhere else
        self.links
            .iter()
            .map(|link| unsafe { link.as_ref() })
            .find(|&node| node.key().borrow() == key)
    }

    /// Find the nearest `Link` or the `Link` which is either the passed in key or just less than the key.
    fn get_nearest<Q>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // There is a unchecked unwrapping but if we some how have a link that points to nothing we messed up somewhere else
        self.links
            .iter()
            .map(|link| unsafe { link.as_ref() })
            .find(|&node| node.key().borrow() <= key)
    }

    /// Find the `Link` which is the passed in key.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut Node<K, V>>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // There is a unchecked unwrapping but if we some how have a link that points to nothing we messed up somewhere else
        self.links
            .iter_mut()
            .map(|link| unsafe { link.as_mut() })
            .find(|node| node.key().borrow() == key)
    }

    /// Find the nearest `Link` or the `Link` which is either the passed in key or just less than the key.
    fn get_nearest_mut<Q>(&mut self, key: &Q) -> Option<&mut Node<K, V>>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // There is a unchecked unwrapping but if we some how have a link that points to nothing we messed up somewhere else
        self.links
            .iter_mut()
            .map(|link| unsafe { link.as_mut() })
            .find(|node| node.key().borrow() <= key)
    }
}

/// A list of `Proxy` structs that link to the underlying linked list.
#[derive(Debug)]
struct ProxyList<K, V> {
    list: Vec<Proxy<K, V>>,
}

impl<K, V> ProxyList<K, V> {
    /// Create a new empty `Self`.
    fn new() -> Self {
        Self {
            list: Default::default(),
        }
    }

    /// Given the level and P value as well as the index in the level return the `Proxy` at the index.
    // TODO: maybe implement index
    fn get_proxy(&self, index: usize) -> &Proxy<K, V> {
        // The index in the proxy list is the passed in index multiplied by the skipping factor (this is just what hte actual index of the element is)
        // divided by the the PROXY_SIZE
        &self.list[(index * P) / PROXY_SIZE]
    }

    /// Given the level and P value as well as the index in the level return the `Proxy` at the index.
    // TODO: maybe implement index
    fn get_proxy_mut(&mut self, index: usize) -> &mut Proxy<K, V> {
        // The index in the proxy list is the passed in index multiplied by the skipping factor (this is just what hte actual index of the element is)
        // divided by the the PROXY_SIZE
        &mut self.list[(index * P) / PROXY_SIZE]
    }
}

impl<K, V> Default for ProxyList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A reference to a fast lane
#[derive(Debug)]
struct Lane<'a, K, V> {
    lane: &'a [Link<K, V>],
}

impl<'a, K, V> Lane<'a, K, V> {
    /// Create a new `Lane`.
    fn new(lane: &'a [Link<K, V>]) -> Self {
        Self { lane }
    }

    /// Get the length of the `Lane`.
    fn len(&self) -> usize {
        self.lane.len()
    }

    /// Find the index in the `Lane` that contains the passed in key or the index - 1 of the value just larger then it.
    /// It is assumed the `Lane` is sorted in ascending order.
    /// It requires the level as well as the previous index so that it knows how far to skip. It needs to skip some elements
    /// based on the result from the previous level searched
    fn find<Q>(&self, key: &Q, level: usize, prev_index: usize) -> usize
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // TODO: Change to something like binary search since this is sorted
        self.lane
            .iter()
            .skip(prev_index * P.pow(level as u32))
            .map(|link| unsafe { link.as_ref() })
            .take_while(|&item| key <= item.key().borrow())
            .count()
    }

    // TODO: Maybe implement index and have a get_unchecked
    /// Get immutable access to the inner slice
    fn inner(&self) -> &[Link<K, V>] {
        self.lane
    }
}

/// Is all the fast lanes used in the `SkipListMap`.
struct FastLanes<K, V> {
    lanes: Vec<Link<K, V>>,
}

impl<K, V> FastLanes<K, V> {
    /// Create a new empty set of `Lanes`.
    fn new() -> Self {
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
    fn lane(&self, level: usize) -> Lane<'_, K, V> {
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

impl<K, V> Default for FastLanes<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Debug for FastLanes<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastLanes")
            .field("lanes", &self.lanes)
            .finish()
    }
}

/// A Cache Sensitive Skip List
#[derive(Debug, Default)]
pub struct SkipListMap<K, V> {
    /// The fast lanes.
    fast_lanes: FastLanes<K, V>,
    /// The proxy list.
    proxy_list: ProxyList<K, V>,
    /// The linked list containing all the nodes.
    data_list: Option<Link<K, V>>,
    /// The number of elements in the `SkipListMap`.
    len: usize,
}

impl<K, V> SkipListMap<K, V> {
    /// Create a new `SkipListMap`.
    pub fn new() -> Self {
        Self {
            fast_lanes: FastLanes::new(),
            proxy_list: Default::default(),
            data_list: None,
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

        let mut index = 0;
        for level in (0..LEVELS).rev() {
            // get the lane that corresponding to that level
            let lane = self.fast_lanes.lane(level);
            // get the index in the lane that is the key or just less than the key
            index = lane.find(key, level, index);
            // check if the index contains the key
            let node = unsafe { lane.inner().get_unchecked(index).as_ref() };
            if node.key().borrow() == key {
                // if it does we can just return the item
                // I don't like this, it makes me sad why must I transmute
                // must I even transmute or am I just dumb
                return Some(unsafe { std::mem::transmute(node.value()) });
            }
            // if not we are going to need to go to the next level
            // but we also need to skip some of the first elements in the next level since
            // the previous level indicates where to start in the next level
        }

        // The key was not found in the fast lanes so we must look at the proxy list based off of the last index
        // which is the index found in the lowest level

        let proxy = self.proxy_list.get_proxy(index);

        // finding the node in the proxy list
        proxy.get(key).map(|node| node.value())
    }

    /// Insert a key value pair into the `SkipListMap`.
    /// Returns the value previously associated with the key and replaces it with the new one if a previous exists.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut index = 0;
        for level in (0..LEVELS).rev() {
            // get the lane that corresponding to that level
            let lane = self.fast_lanes.lane(level);
            // get the index in the lane that is the key or just less than the key
            index = lane.find(&key, level, index);
            // check if the index contains the key
            let node = unsafe { lane.inner().get_unchecked(index) };

            // TODO: this worries me since this seems unsafe, I should maybe change this
            let node = unsafe { &mut *node.as_ptr() };
            if node.key().borrow() == &key {
                // if it does we can just update the item
                return Some(node.item_mut().replace(value));
            }
            // if not we are going to need to go to the next level
            // but we also need to skip some of the first elements in the next level since
            // the previous level indicates where to start in the next level
        }

        // The key was not found in the fast lanes so we must look at the proxy list based off of the last index
        // which is the index found in the lowest level

        let proxy = self.proxy_list.get_proxy_mut(index);

        // finding the nearest node in the proxy list
        let node = proxy.get_nearest_mut(&key);

        if let Some(node) = node {
            // if there is a node first check if they have the same key
            if node.key().borrow() == &key {
                // if so replace the value
                Some(node.item_mut().replace(value))
            } else {
                // else we append the new node infront of it
                let new_node = Node::new(key, value);
                node.append(new_node);
                None
            }
        } else {
            // if there is no node it means this node belongs in the front of the linked list
            // I think at least still need to properly think about this and confirm it
            let mut new_node = Node::new(key, value);
            new_node.append(self.data_list);
            None
        }
    }
}
