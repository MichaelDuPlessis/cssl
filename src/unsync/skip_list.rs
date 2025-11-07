use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, ptr::NonNull};

/// Calculates the lane start given the total number of elements in the fast lanes combined, the P value and lane level.
/// The level is 0 indexed.
// TODO: Add derivation for calculation
fn lane_start(level: u8, levels: u8, p: u8, total_elements: usize) -> usize {
    // TODO: This can be transformed from a loop to a single sum
    (level + 1..levels)
        .map(|l| lane_len(l, p, total_elements))
        .sum()
}

/// Calculates the total number of elements in a lane.
fn lane_len(level: u8, p: u8, total_elements: usize) -> usize {
    // remember we start 0 indexed
    let pl = p.pow(level as u32 + 1) as usize;
    (total_elements + pl - 1) / pl
}

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

    /// Retrieve mutable a reference to the value.
    fn value_mut(&mut self) -> &mut V {
        &mut self.value
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
    fn append(&mut self, link: impl Into<Link<K, V>>) {
        let mut link = link.into();
        unsafe { link.as_mut() }.next = self.next;
        self.next = Some(link);
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

    /// Get a mutable refernce to the value.
    fn value_mut(&mut self) -> &mut V {
        self.item.value_mut()
    }

    /// Checks if the passed in key matches the key of the Node.
    fn key_matches<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        key == self.key().borrow()
    }

    /// Looks for the provided key in the `Node` and its successors and returns a reference to the `Node` with that key or returns the node just before if it exists.
    fn find<Q>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.iter()
            .take_while(|&item| key >= item.key().borrow())
            .last()
    }

    /// Looks for the provided key in the `Node` and its successors and returns a reference to the `Node` with that key or returns the node just before if it exists.
    fn find_mut<Q>(&mut self, key: &Q) -> Option<&mut Node<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.iter_mut()
            .take_while(|item| key >= item.key().borrow())
            .last()
    }

    /// Get a pointer to the next node if there is one
    fn next(&self) -> Option<Link<K, V>> {
        self.next
    }

    /// Create an iterator over immutable references
    fn iter(&self) -> NodeIter<'_, K, V> {
        NodeIter {
            current: Some(self.into()),
            _marker: PhantomData,
        }
    }

    /// Create an iterator over mutable references
    fn iter_mut(&mut self) -> NodeIterMut<'_, K, V> {
        NodeIterMut {
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

impl<K, V> From<Node<K, V>> for Link<K, V> {
    fn from(value: Node<K, V>) -> Self {
        unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) }
    }
}

/// Immutabale iterator for `Node`.
struct NodeIter<'a, K: 'a, V: 'a> {
    current: Option<Link<K, V>>,
    _marker: PhantomData<&'a Link<K, V>>,
}

impl<'a, K, V> Iterator for NodeIter<'a, K, V> {
    type Item = &'a Node<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.current?; // copy out the link (Link<T> is Copy)
        let node = unsafe { link.as_ref() };

        // Advance the iterator
        self.current = node.next();

        Some(node)
    }
}

impl<'a, K, V> IntoIterator for &'a Node<K, V> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = NodeIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutable iterator for `Node`.
struct NodeIterMut<'a, K: 'a, V: 'a> {
    current: Option<Link<K, V>>,
    _marker: PhantomData<&'a mut Link<K, V>>,
}

impl<'a, K, V> Iterator for NodeIterMut<'a, K, V> {
    type Item = &'a mut Node<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut link = self.current?; // copy out the link (Link<T> is Copy)
        let node = unsafe { link.as_mut() };

        // Advance the iterator
        self.current = node.next;

        Some(node)
    }
}

impl<'a, K, V> IntoIterator for &'a mut Node<K, V> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = NodeIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// A reference to a fast lane.
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

    /// Find the index in the `Lane` that contains the passed in key or the index - 1 of the value just larger then it otherwise None.
    /// It is assumed the `Lane` is sorted in ascending order.
    /// It requires the level as well as the previous index so that it knows how far to skip. It needs to skip some elements
    /// based on the result from the previous level searched
    fn find<Q>(&self, key: &Q, level: u8, p: u8, prev_index: usize) -> Option<usize>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        let skip_count = prev_index * p.pow(level as u32 + 1) as usize;
        let search_slice = &self.lane[skip_count..];
        
        let result = search_slice.binary_search_by(|link| {
            let node = unsafe { link.as_ref() };
            node.key().borrow().cmp(key)
        });
        
        match result {
            Ok(exact_index) => Some(exact_index),
            Err(insert_index) => insert_index.checked_sub(1),
        }
    }

    // TODO: Maybe implement index and have a get_unchecked
    /// Get immutable access to the inner slice
    fn inner(&self) -> &[Link<K, V>] {
        self.lane
    }
}

impl<'a, K, V> Debug for Lane<'a, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane").field("lane", &self.lane).finish()
    }
}

/// Is all the fast lanes used in the `SkipListMap`.
struct FastLanes<K, V> {
    /// The number of levels
    levels: u8,
    /// The fast lanes
    lanes: Vec<Link<K, V>>,
    /// The number of elements when the fast lane was built
    num_elements: usize,
}

impl<K, V> FastLanes<K, V> {
    /// Create a new empty set of `Lanes`.
    fn new(levels: u8) -> Self {
        Self {
            levels,
            lanes: Vec::new(),
            num_elements: 0,
        }
    }

    /// Get lane start and length for a given level
    fn lane_bounds(&self, level: u8, levels: u8, p: u8) -> (usize, usize) {
        let total_elements = self.num_elements;
        let start = lane_start(level, levels, p, total_elements);
        let len = lane_len(level, p, total_elements);
        (start, len)
    }

    /// Retrieve a reference to the fast lane located on the nth level.
    fn lane(&self, level: u8, levels: u8, p: u8) -> Lane<'_, K, V> {
        if self.lanes.is_empty() {
            return Lane::new(&[]);
        }

        let (start, len) = self.lane_bounds(level, levels, p);
        Lane::new(&self.lanes[start..start + len])
    }

    /// Remove the specified key from all lanes and returns a link to the `Node` just before the remove key.
    fn remove<Q>(&mut self, key: &Q, levels: u8, p: u8) -> Option<Link<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let total_elements = self.num_elements;

        // find element in lowest lane
        let lane = self.lane(0, levels, p);
        let index = lane.find(key, 0, p, 0)?;

        // if the link we found is not the one we are looking for it means that the link is not in the fast lanes
        // so just return closest element to the one to be removed
        let link = *unsafe { lane.inner().get_unchecked(index) };

        // if it doesn't match it means that the key is not in the fast lanes and we can just return this link
        if !unsafe { link.as_ref().key_matches(key) } {
            return Some(link);
        }

        // Update level 0
        let (start, len) = self.lane_bounds(0, levels, p);
        let abs_index = start + index;

        if len == 1 {
            // if the lanes length is one and if we got here the key matches the link so we can just clear all the fast lanes
            self.lanes.clear();
            return None;
        }

        if index < len - 1 {
            self.lanes[abs_index] = self.lanes[abs_index + 1];
        } else {
            self.lanes[abs_index] = self.lanes[abs_index - 1];
        }

        // Update higher levels
        for level in 1..levels {
            let lane = self.lane(level, levels, p);
            // TODO: I should know the previous index to start looking from
            if let Some(level_index) = lane.find(key, level, p, 0) {
                let link = unsafe { lane.inner().get_unchecked(level_index) };

                if unsafe { link.as_ref().key_matches(key) } {
                    // if we found the link
                    let (start, len) = self.lane_bounds(level, levels, p);
                    let abs_index = start + level_index;

                    self.lanes[abs_index] = if len == 1 {
                        // Use element from lower lane
                        let (lower_start, _) = self.lane_bounds(level - 1, levels, p);
                        self.lanes[lower_start]
                    } else if level_index < len - 1 {
                        self.lanes[abs_index + 1]
                    } else {
                        self.lanes[abs_index - 1]
                    }
                } else {
                    // if it was not found on this level it cannot be on higher levels since lower levels contain
                    // all the values on higher levels
                    break;
                }
            } else {
                // if it was not found on this level it cannot be on higher levels since lower levels contain
                // all the values on higher levels
                break;
            }
        }

        if index == 0 {
            None
        } else {
            self.lanes.get(abs_index - 1).copied()
        }
    }

    /// The number of fast lanes
    fn num_lanes(&self) -> u8 {
        self.levels
    }
}

impl<K, V> Default for FastLanes<K, V> {
    fn default() -> Self {
        Self::new(2)
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
#[derive(Debug)]
pub struct SkipListMap<K, V> {
    /// The fast lanes.
    fast_lanes: FastLanes<K, V>,
    /// The linked list containing all the nodes.
    data_list: Option<Link<K, V>>,
    /// The number of elements in the `SkipListMap`.
    len: usize,
    /// The P value for the `SkipList`, this is how many elements are skipped going up levels of the fast lane.
    /// This will be known as the skipping factor
    p: u8,
}

impl<K, V> SkipListMap<K, V> {
    /// Create a new `SkipListMap`.
    pub fn new(p: u8, levels: u8) -> Self {
        Self {
            fast_lanes: FastLanes::new(levels),
            data_list: None,
            len: 0,
            p,
        }
    }

    /// The number of elements in the `SkipListMap`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the skip list is empty otherwise false.
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the number of fast lanes.
    fn levels(&self) -> u8 {
        self.fast_lanes.levels
    }

    /// Builds the fast lanes from the data list.
    fn build_fast_lanes(&mut self) {
        // there is nothing to build if there are no elements
        if self.is_empty() {
            return;
        }
        let levels = self.levels();

        // first calculate the needed size for the fast lanes
        // TODO: This can be done better no loops should be needed
        let len = (0..levels).map(|i| lane_len(i, self.p, self.len())).sum();
        let mut lanes = vec![NonNull::dangling(); len];

        // calculating lane offsets using the consolidated method
        let offsets: Vec<usize> = (0..levels)
            .map(|level| lane_start(level, levels, self.p, self.len()))
            .collect();

        if let Some(link) = self.data_list {
            let node = unsafe { link.as_ref() };

            for (i, node) in node.iter().enumerate() {
                for level in 1..=levels {
                    let p = self.p.pow(level as u32) as usize;
                    if i % p == 0 {
                        let index = i / p + offsets[level as usize - 1];
                        lanes[index] = node.into();
                    }
                }
            }
        }

        self.fast_lanes = FastLanes {
            levels,
            lanes,
            num_elements: self.len(),
        };
    }

    /// Safely remove a node and extract its value
    unsafe fn remove_node(node_ptr: Link<K, V>) -> V {
        let raw_ptr = node_ptr.as_ptr();
        unsafe {
            let node_moved: Node<K, V> = std::ptr::read(raw_ptr);
            let mut node_moved = std::mem::ManuallyDrop::new(node_moved);
            let val = std::ptr::read(&mut node_moved.item.value);
            let layout = std::alloc::Layout::new::<Node<K, V>>();
            std::alloc::dealloc(raw_ptr as *mut u8, layout);
            val
        }
    }

    /// Create an immutable iterator.
    fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            current: self.data_list,
            _marker: PhantomData,
        }
    }

    /// Create an mutable iterator.
    fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            current: self.data_list,
            _marker: PhantomData,
        }
    }
}

impl<K, V> SkipListMap<K, V>
where
    K: Ord,
{
    /// Find the starting node for a key search using fast lanes or the node just before the searched for node
    /// if it exists
    fn find_start_node_impl<Q>(&self, key: &Q) -> Option<Link<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut index = 0;

        // Search through fast lanes from highest to lowest level
        let levels = self.levels();
        for level in (1..levels).rev() {
            let lane = self.fast_lanes.lane(level, levels, self.p);
            if let Some(possible_index) = lane.find(key, level, self.p, index) {
                index = possible_index;
                let node_link = *unsafe { lane.inner().get_unchecked(index) };
                let node = unsafe { node_link.as_ref() };
                if node.key_matches(key) {
                    return Some(node_link);
                }
            }
        }

        // Search level 0
        let lane = self.fast_lanes.lane(0, levels, self.p);
        if let Some(index) = lane.find(key, 0, self.p, index) {
            let node_link = *unsafe { lane.inner().get_unchecked(index) };
            Some(node_link)
        } else {
            self.data_list
        }
    }

    /// Find the starting node for a key search using fast lanes or the node just before the searched for node
    /// if it exists
    fn find_start_node<Q>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_start_node_impl(key).map(|link| unsafe { link.as_ref() })
    }

    /// Find the starting mutable node for a key search using fast lanes or the node just before the searched node if it exists
    fn find_start_node_mut<Q>(&mut self, key: &Q) -> Option<&mut Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_start_node_impl(key).map(|mut link| unsafe { link.as_mut() })
    }

    /// Retrieves an item from the `SkipListMap`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let start_node = self.find_start_node(key)?;

        if start_node.key_matches(key) {
            return Some(start_node.value());
        }

        let node = start_node.find(key)?;
        if node.key_matches(key) {
            Some(node.value())
        } else {
            None
        }
    }

    /// Insert a key value pair into the `SkipListMap`.
    /// Returns the value previously associated with the key and replaces it with the new one if a previous exists.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Handle empty list case
        if self.data_list.is_none() {
            let new_node = Node::new(key, value);
            self.len += 1;
            self.data_list = new_node.into();
            return None;
        }

        let start_node = self.find_start_node_mut(&key).unwrap();

        // Check if start node matches
        if start_node.key_matches(&key) {
            return Some(start_node.item_mut().replace(value));
        }

        // Search for exact match or insertion point
        if let Some(node) = start_node.find_mut(&key) {
            if node.key_matches(&key) {
                Some(node.item_mut().replace(value))
            } else {
                node.append(Node::new(key, value));
                self.len += 1;
                None
            }
        } else {
            // Insert at head
            let mut new_node = Node::new(key, value);
            new_node.next = self.data_list;
            self.data_list = new_node.into();
            self.len += 1;
            None
        }
    }

    /// Removes the key fromthe `SkipListMap` if it exists and returns the value associated with it.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let levels = self.levels();
        let prev_link = self.fast_lanes.remove(key, levels, self.p);
        let mut start = prev_link.unwrap_or(self.data_list?);

        let head = unsafe { start.as_ref() };

        // this means the node to remove is the head since there is no previous link
        if prev_link.is_none() && head.key_matches(key) {
            let node_ptr: Link<K, V> = head.into();
            self.data_list = head.next();
            self.len -= 1;
            return Some(unsafe { Self::remove_node(node_ptr) });
        }

        let mut previous = start;
        for node_ref in unsafe { start.as_mut() }.iter_mut().skip(1) {
            if node_ref.key_matches(key) {
                let node_ptr: Link<K, V> = node_ref.into();
                unsafe { previous.as_mut().next = node_ref.next };
                self.len -= 1;
                return Some(unsafe { Self::remove_node(node_ptr) });
            }

            if node_ref.key().borrow() > key {
                break;
            }

            previous = node_ref.into();
        }

        None
    }
}

impl<K, V> Default for SkipListMap<K, V> {
    fn default() -> Self {
        Self::new(2, 2)
    }
}

impl<K, V> Drop for SkipListMap<K, V> {
    fn drop(&mut self) {
        let mut current = self.data_list;
        while let Some(cur) = current {
            unsafe {
                let node = Box::from_raw(cur.as_ptr());
                current = node.next;
            }
        }
    }
}

// TODO: I think this is unnecessary since we can just use Node iter
/// Immutabale iterator for `SkipListMap`.
pub struct Iter<'a, K: 'a, V: 'a> {
    current: Option<Link<K, V>>,
    _marker: PhantomData<&'a Link<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.current?; // copy out the link (Link<T> is Copy)
        let node = unsafe { link.as_ref() };

        // Advance the iterator
        self.current = node.next;

        Some((node.key(), node.value()))
    }
}

impl<'a, K, V> IntoIterator for &'a SkipListMap<K, V> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutabale iterator for `SkipListMap`.
pub struct IterMut<'a, K: 'a, V: 'a> {
    current: Option<Link<K, V>>,
    _marker: PhantomData<&'a Link<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.current?; // copy out the link (Link<T> is Copy)
        let node = link.as_ptr();

        // Advance the iterator
        self.current = (unsafe { &*node }).next;

        Some((
            (unsafe { &*node }).key(),
            (unsafe { &mut *node }).value_mut(),
        ))
    }
}

impl<'a, K, V> IntoIterator for &'a mut SkipListMap<K, V> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insertion() {
        const NUM_VALUES: usize = 32;

        let mut map = SkipListMap::default();
        for i in 0..NUM_VALUES {
            assert_eq!(None, map.insert(i, i));
        }

        assert_eq!(map.len(), NUM_VALUES);

        for (i, (k, v)) in (0..NUM_VALUES).zip(map.iter()) {
            assert_eq!(i, *k);
            assert_eq!(i, *v);
        }

        assert_eq!(map.get(&8), Some(&8));
        assert_eq!(map.get(&31), Some(&31));
        assert_eq!(map.get(&12), Some(&12));
        assert_eq!(map.get(&32), None);

        for i in 0..NUM_VALUES {
            assert_eq!(Some(i), map.insert(i, i + 1));
        }
        assert_eq!(map.len(), NUM_VALUES);
    }

    #[test]
    fn basic_deletion() {
        const NUM_VALUES: usize = 32;

        let mut map = SkipListMap::default();
        for i in 0..NUM_VALUES {
            assert_eq!(None, map.insert(i, i));
        }

        assert_eq!(map.len(), NUM_VALUES);

        assert_eq!(map.remove(&8), Some(8));
        assert_eq!(map.remove(&31), Some(31));
        assert_eq!(map.remove(&12), Some(12));
        assert_eq!(map.remove(&12), None);
        assert_eq!(map.remove(&32), None);

        assert_eq!(map.len(), NUM_VALUES - 3);

        assert_eq!(map.get(&8), None);
        assert_eq!(map.get(&13), Some(&13));
    }

    #[test]
    fn fast_lane() {
        const NUM_VALUES: usize = 32;

        let mut map = SkipListMap::default();
        for i in 0..NUM_VALUES {
            assert_eq!(None, map.insert(i, i));
        }

        map.build_fast_lanes();

        assert_eq!(map.get(&8), Some(&8));
        assert_eq!(map.get(&31), Some(&31));
        assert_eq!(map.get(&12), Some(&12));
        assert_eq!(map.get(&32), None);

        assert_eq!(map.remove(&8), Some(8));

        assert_eq!(map.get(&8), None);
        assert_eq!(None, map.insert(8, 8));

        for i in 0..NUM_VALUES {
            assert_eq!(Some(i), map.insert(i, i + 1));
        }
        assert_eq!(map.get(&12), Some(&13));
    }

    #[test]
    fn edge_cases() {
        let mut map = SkipListMap::default();
        assert_eq!(None, map.insert(0, 0));
        map.build_fast_lanes();

        // now there should only be two elements across all fast lanes
        assert_eq!(map.fast_lanes.lanes.len(), 2);

        // what happens if we removeit
        map.remove(&0);

        let mut map = SkipListMap::default();
        assert_eq!(None, map.insert(0, 0));
        assert_eq!(None, map.insert(1, 0));
        assert_eq!(None, map.insert(2, 0));
        map.build_fast_lanes();

        assert_eq!(map.fast_lanes.lanes.len(), 3);

        assert_eq!(Some(0), map.remove(&0));

        assert_eq!(None, map.insert(0, 0));
        assert_eq!(Some(&0), map.get(&0));
        assert_eq!(Some(0), map.remove(&0));
        assert_eq!(None, map.get(&0));
    }

    #[test]
    fn empty_map_operations() {
        let mut map = SkipListMap::<i32, i32>::default();

        // Operations on empty map
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.remove(&1), None);

        // Iterator on empty map
        assert_eq!(map.iter().count(), 0);
        assert_eq!(map.iter_mut().count(), 0);
    }

    #[test]
    fn single_element_operations() {
        let mut map = SkipListMap::default();

        // Insert single element
        assert_eq!(map.insert(42, "answer"), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42), Some(&"answer"));

        // Replace single element
        assert_eq!(map.insert(42, "new_answer"), Some("answer"));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42), Some(&"new_answer"));

        // Remove single element
        assert_eq!(map.remove(&42), Some("new_answer"));
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&42), None);

        // Remove from empty map after removal
        assert_eq!(map.remove(&42), None);
    }

    #[test]
    fn head_operations() {
        let mut map = SkipListMap::default();

        // Insert elements in ascending order
        for i in 0..5 {
            map.insert(i, i * 10);
        }

        // Remove head element
        assert_eq!(map.remove(&0), Some(0));
        assert_eq!(map.get(&0), None);
        assert_eq!(map.get(&1), Some(&10));

        // Insert new head
        assert_eq!(map.insert(-1, -10), None);
        assert_eq!(map.get(&-1), Some(&-10));

        // Remove new head
        assert_eq!(map.remove(&-1), Some(-10));
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn tail_operations() {
        let mut map = SkipListMap::default();

        for i in 0..5 {
            map.insert(i, i * 10);
        }

        // Remove tail element
        assert_eq!(map.remove(&4), Some(40));
        assert_eq!(map.get(&4), None);
        assert_eq!(map.get(&3), Some(&30));

        // Insert new tail
        assert_eq!(map.insert(10, 100), None);
        assert_eq!(map.get(&10), Some(&100));
    }

    #[test]
    fn duplicate_keys() {
        let mut map = SkipListMap::default();

        // Insert same key multiple times
        assert_eq!(map.insert(1, "first"), None);
        assert_eq!(map.insert(1, "second"), Some("first"));
        assert_eq!(map.insert(1, "third"), Some("second"));

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"third"));

        // Remove should return latest value
        assert_eq!(map.remove(&1), Some("third"));
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn reverse_order_insertion() {
        let mut map = SkipListMap::default();

        // Insert in reverse order
        for i in (0..10).rev() {
            map.insert(i, i * 2);
        }

        // Verify order is maintained
        let collected: Vec<_> = map.iter().collect();
        for (i, (k, v)) in collected.iter().enumerate() {
            assert_eq!(**k, i);
            assert_eq!(**v, i * 2);
        }
    }

    #[test]
    fn random_order_operations() {
        let mut map = SkipListMap::new(2, 4);
        let keys = [5, 2, 8, 1, 9, 3, 7, 4, 6];

        // Insert in random order
        for &key in &keys {
            map.insert(key, key * 10);
            assert_eq!(map.get(&5), Some(&50));
        }

        assert_eq!(map.len(), 9);
        assert_eq!(map.get(&5), Some(&50));

        // Verify sorted iteration
        let collected: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        let expected: Vec<_> = (1..=9).map(|i| (i, i * 10)).collect();
        assert_eq!(collected, expected);

        // Remove in different order
        assert_eq!(map.remove(&5), Some(50));
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.remove(&9), Some(90));

        assert_eq!(map.len(), 6);
    }

    #[test]
    fn fast_lanes_with_removals() {
        let mut map = SkipListMap::default();

        // Insert many elements
        for i in 0..20 {
            map.insert(i, i);
        }

        map.build_fast_lanes();

        // Remove elements from different positions
        assert_eq!(map.remove(&0), Some(0)); // head
        assert_eq!(map.remove(&19), Some(19)); // tail
        assert_eq!(map.remove(&10), Some(10)); // middle

        // Verify remaining elements
        assert_eq!(map.len(), 17);
        assert_eq!(map.get(&0), None);
        assert_eq!(map.get(&19), None);
        assert_eq!(map.get(&10), None);
        assert_eq!(map.get(&5), Some(&5));
        assert_eq!(map.get(&15), Some(&15));
    }

    #[test]
    fn iterator_consistency() {
        let mut map = SkipListMap::default();

        for i in [3, 1, 4, 1, 5, 9, 2, 6] {
            map.insert(i, i * i);
        }

        // Immutable iterator
        let keys: Vec<_> = map.iter().map(|(&k, _)| k).collect();
        let values: Vec<_> = map.iter().map(|(_, &v)| v).collect();

        // Should be sorted by key
        assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 9]);
        assert_eq!(values, vec![1, 4, 9, 16, 25, 36, 81]);

        // Mutable iterator
        for (_, v) in map.iter_mut() {
            *v += 1;
        }

        let new_values: Vec<_> = map.iter().map(|(_, &v)| v).collect();
        assert_eq!(new_values, vec![2, 5, 10, 17, 26, 37, 82]);
    }

    #[test]
    fn large_dataset() {
        let mut map = SkipListMap::new(5, 2);
        const SIZE: usize = 1000;

        // Insert large dataset
        for i in 0..SIZE {
            map.insert(i, i * 3);
        }

        map.build_fast_lanes();

        // Random access
        assert_eq!(map.get(&500), Some(&1500));
        assert_eq!(map.get(&0), Some(&0));
        assert_eq!(map.get(&999), Some(&2997));
        assert_eq!(map.get(&1000), None);

        // Remove every 10th element
        for i in (0..SIZE).step_by(10) {
            assert_eq!(map.remove(&i), Some(i * 3));
        }

        assert_eq!(map.len(), SIZE - SIZE / 10);

        // Verify removed elements are gone
        for i in (0..SIZE).step_by(10) {
            assert_eq!(map.get(&i), None);
        }

        // Verify remaining elements
        for i in 1..SIZE {
            if i % 10 != 0 {
                assert_eq!(map.get(&i), Some(&(i * 3)));
            }
        }
    }

    #[test]
    fn boundary_value_tests() {
        let mut map = SkipListMap::default();

        // Test with extreme values
        map.insert(i32::MIN, "min");
        map.insert(i32::MAX, "max");
        map.insert(0, "zero");
        map.insert(-1, "neg_one");
        map.insert(1, "one");

        assert_eq!(map.get(&i32::MIN), Some(&"min"));
        assert_eq!(map.get(&i32::MAX), Some(&"max"));
        assert_eq!(map.get(&0), Some(&"zero"));

        // Verify order
        let keys: Vec<_> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(keys, vec![i32::MIN, -1, 0, 1, i32::MAX]);
    }
}
