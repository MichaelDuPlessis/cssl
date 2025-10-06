use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, ptr::NonNull};

/// The P value for the `SkipList`, this is how many elements are skipped going up levels of the fast lane.
/// This will be known as the skipping factor
const P: usize = 2;

/// The number of fastlane levels in the `SkipList`.
const LEVELS: usize = 2;

/// Calculates the lane start given the total number of elements in the fast lanes combined, the P value and lane level.
/// The level is 0 indexed.
// TODO: Add derivation for calculation
fn lane_start(p: usize, level: usize, total_elements: usize) -> usize {
    if level == LEVELS - 1 {
        return 0;
    }

    let level = level as u32 + 1;
    (total_elements * (p.pow(level) - 1)) / (p.pow(level + 1) * (p - 1))
}

/// Calculates the total number of elements in a lane.
fn lane_len(level: usize, total_elements: usize) -> usize {
    // remember we start 0 indexed
    total_elements / P.pow(level as u32 + 1)
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

    /// Checks if the passed in key matches the key of the Node.
    fn key_matches<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        key == self.key().borrow()
    }

    /// Looks for the provided key in the `Node` and its successors and returns a reference to the `Node` with that key.
    fn find<Q>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.iter()
            .take_while(|&item| key >= item.key().borrow())
            .last()
    }

    /// Looks for the provided key in the `Node` and its successors and returns a reference to the `Node` with that key.
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

    /// Find the index in the `Lane` that contains the passed in key or the index - 1 of the value just larger then it otherwise None.
    /// It is assumed the `Lane` is sorted in ascending order.
    /// It requires the level as well as the previous index so that it knows how far to skip. It needs to skip some elements
    /// based on the result from the previous level searched
    fn find<Q>(&self, key: &Q, level: usize, prev_index: usize) -> Option<usize>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        // TODO: Change to something like binary search since this is sorted
        let index = self
            .lane
            .iter()
            .skip(prev_index * P.pow(level as u32 + 1))
            .map(|link| unsafe { link.as_ref() })
            .take_while(|&node| key >= node.key().borrow())
            .count();

        index.checked_sub(1)
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
    fn lane_len(&self, level: usize, total_elements: usize) -> usize {
        lane_len(level, total_elements)
    }

    fn lane_start(&self, level: usize, total_elements: usize) -> usize {
        // The start of a level is the sum of the lengths of the previous levels
        // taking the formula for the level_len summing it an simplifying leads to an
        // equation with no loops, aint that cool
        // remember we start 0 indexed

        lane_start(P, level, total_elements)
    }

    /// Retrieve a reference to the fast lane located on the nth level.
    fn lane(&self, level: usize, total_elements: usize) -> Lane<'_, K, V> {
        if self.lanes.is_empty() {
            return Lane::new(&[]);
        }

        // if it is the highest level we start at
        // Calculating the beginning and end of the level
        // 1. calculate the number of elements in the level
        let num_elements = self.lane_len(level, total_elements);

        // 2. find the start of the level
        let lane_start = self.lane_start(level, total_elements);

        Lane::new(&self.lanes[lane_start..lane_start + num_elements])
    }

    /// Remove the specified key from all lanes and returns a link to the `Node` just before the remove key.
    fn remove<Q>(&mut self, key: &Q, total_elements: usize) -> Option<Link<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        // get the previous element for lane 0 as if there isn't one the upper lanes will also not have one
        let lane = self.lane(0, total_elements);
        let index = lane.find(key, 0, 0);

        if let Some(index) = index {
            // first check if the index found matches the key
            let link = *unsafe { lane.inner().get_unchecked(index) };
            if unsafe { link.as_ref().key_matches(key) } {
                let abs_index = self.lane_start(0, total_elements) + index;
                if index < lane.len() - 1 {
                    // check if the lane is long enough to copy next element
                    self.lanes[abs_index] = self.lanes[abs_index + 1];
                } else {
                    // if we are at the end we must copy from the other way
                    self.lanes[abs_index] = self.lanes[abs_index - 1];
                }

                // if the key is in the lowest lane it can exist in a higher lane
                for level in 1..LEVELS {
                    let lane = self.lane(level, total_elements);

                    // TODO: Since the index of the lower element is known we can skip ahead here
                    let index = lane.find(key, level, 0);
                    if let Some(index) = index {
                        // first check if the index found matches the key
                        let link = unsafe { lane.inner().get_unchecked(index) };
                        if unsafe { link.as_ref().key_matches(key) } {
                            let abs_index = self.lane_start(level, total_elements) + index;
                            if index < lane.len() - 1 {
                                // check if the lane is long enough to copy next element
                                self.lanes[abs_index] = self.lanes[abs_index + 1];
                            } else {
                                // if we are at the end we must copy from the other way
                                self.lanes[abs_index] = self.lanes[abs_index - 1];
                            }
                        }
                    }
                }

                self.lanes.get(abs_index.checked_sub(1)?).copied()
            } else {
                // this is a link before the one we are looking for
                Some(unsafe { *lane.inner().get_unchecked(index) })
            }
        } else {
            None
        }
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
            data_list: None,
            len: 0,
        }
    }

    /// The number of elements in the `SkipListMap`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Builds the fast lanes from the data list.
    fn build_fast_lanes(&mut self) {
        // first calculate the needed size for the fast lanes
        let pn = P.pow(LEVELS as u32);
        let len = (self.len() * (pn - 1)) / (P.pow(LEVELS as u32 + 1) - pn);
        let mut lanes: Vec<Link<K, V>> = Vec::with_capacity(len);
        // set the len so we can just index and assign anywhere
        unsafe { lanes.set_len(len) };

        // calculating lane offsets
        // TODO: This is essentially a copy of the code from the FastLane struct and should be extracted out
        let mut offsets = [0; LEVELS];
        for level in 0..LEVELS {
            let offset = lane_start(P, level, self.len());
            offsets[level] = offset;
        }

        if let Some(link) = self.data_list {
            let node = unsafe { link.as_ref() };

            for (i, node) in node.iter().enumerate() {
                for level in 1..=LEVELS {
                    let p = P.pow(level as u32);
                    if i % p == 0 {
                        let index = i / p + offsets[level - 1];
                        lanes[index] = node.into();
                    }
                }
            }
        }

        self.fast_lanes = FastLanes { lanes };
    }

    /// Create an immutable iterator.
    fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            current: self.data_list,
            _marker: PhantomData,
        }
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
        let mut index = 0;
        for level in (1..LEVELS).rev() {
            // get the lane that corresponding to that level
            let lane = self.fast_lanes.lane(level, self.len());
            // get the index in the lane that is the key or just less than the key
            let possible_index = lane.find(key, level, index);

            if let Some(possible_index) = possible_index {
                index = possible_index;

                // check if the index contains the key
                let node = unsafe { lane.inner().get_unchecked(index).as_ref() };
                if node.key_matches(key) {
                    // if it does we can just return the item
                    // I don't like this, it makes me sad why must I transmute
                    // must I even transmute or am I just dumb
                    return Some(unsafe { std::mem::transmute(node.value()) });
                }
            }
            // if not we are going to need to go to the next level
            // but we also need to skip some of the first elements in the next level since
            // the previous level indicates where to start in the next level
        }

        // this is just a repeat of the top logic
        let lane = self.fast_lanes.lane(0, self.len());
        let index = lane.find(key, 0, index);
        let node = if let Some(index) = index {
            unsafe { lane.inner().get_unchecked(index).as_ref() }
        } else {
            unsafe { self.data_list?.as_ref() }
        };

        // the key was not found in the fast lanes so now we must iterate over the node list until the key is found or a larger is found in which case
        // it means the key does not exist
        let node = node.find(key)?;

        // now we need to check if the key was found
        if node.key_matches(key) {
            Some(node.value())
        } else {
            None
        }
    }

    /// Insert a key value pair into the `SkipListMap`.
    /// Returns the value previously associated with the key and replaces it with the new one if a previous exists.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut index = 0;
        for level in (1..LEVELS).rev() {
            // get the lane that corresponding to that level
            let lane = self.fast_lanes.lane(level, self.len());
            // get the index in the lane that is the key or just less than the key
            let possible_index = lane.find(&key, level, index);

            if let Some(possible_index) = possible_index {
                index = possible_index;

                // check if the index contains the key
                let node = unsafe { lane.inner().get_unchecked(index) };

                // TODO: this worries me since this seems unsafe, I should maybe change this
                let node = unsafe { &mut *node.as_ptr() };
                if node.key_matches(&key) {
                    // if it does we can just update the item
                    return Some(node.item_mut().replace(value));
                }
            }
            // if not we are going to need to go to the next level
            // but we also need to skip some of the first elements in the next level since
            // the previous level indicates where to start in the next level
        }

        // this is just a repeat of the top logic
        let lane = self.fast_lanes.lane(0, self.len());
        let index = lane.find(&key, 0, index);
        let node = if let Some(index) = index {
            let node = unsafe { lane.inner().get_unchecked(index) };
            // TODO: same as above todo
            unsafe { &mut *node.as_ptr() }
        } else {
            if let Some(mut link) = self.data_list {
                unsafe { link.as_mut() }
            } else {
                // there is no head lets make one
                self.len += 1;
                self.data_list = Node::new(key, value).into();
                return None;
            }
        };

        // the key was not found in the fast lanes so now we must iterate over the node list until the key is found or a larger is found in which case
        // it means the key does not exist
        let node = node.find_mut(&key);

        if let Some(node) = node {
            // if there is a node first check if they have the same key
            if node.key_matches(&key) {
                // if so replace the value
                Some(node.item_mut().replace(value))
            } else {
                self.len += 1;
                // else we append the new node infront of it
                let new_node = Node::new(key, value);
                node.append(new_node);
                None
            }
        } else {
            self.len += 1;
            // if there is no node it means this node belongs in the front of the linked list
            // I think at least still need to properly think about this and confirm it
            let mut new_node = Node::new(key, value);
            if let Some(link) = self.data_list {
                new_node.append(link);
            }
            self.data_list = new_node.into();
            None
        }
    }

    /// Removes the key fromthe `SkipListMap` if it exists and returns the value associated with it.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // first remove the element from every lane if it exists.
        let prev_link = self.fast_lanes.remove(key, self.len());

        // start searching from prev_link (if present) or head otherwise
        let start = if let Some(prev) = prev_link {
            prev
        } else {
            // if there is no previous link then start from head
            self.data_list?
        };

        // check if head node contains the value
        let head = unsafe { start.as_ref() };
        if prev_link.is_none() && head.key_matches(key) {
            // node_ptr is NonNull<Node<K,V>> pointing to the node to remove
            let node_ptr: Link<K, V> = head.into();

            // 1) Link previous -> node.next (unlink)
            self.data_list = head.next();

            // 2) Now reclaim ownership of the node's allocation and extract the value.
            //
            // We used ptr::read to move the Node out of heap memory into a stack value,
            // then wrap it in ManuallyDrop so Drop does not run for that stack copy.
            // Finally we extract the value V using ptr::read and deallocate the original
            // heap allocation with the global allocator. This avoids double-drop.
            let value = unsafe {
                let raw_ptr = node_ptr.as_ptr();

                // Move the Node out of the allocation (bitwise move, does not call drop).
                let node_moved: Node<K, V> = std::ptr::read(raw_ptr);
                let mut node_moved = std::mem::ManuallyDrop::new(node_moved);

                // Extract the value (move out) from the Item inside the node.
                // Using ptr::read to avoid running Drop for the moved-out field later.
                let val = std::ptr::read(&mut node_moved.item.value);

                // We've already unlinked the node from the list; the heap allocation
                // still needs to be deallocated. Because we used std::ptr::read to
                // create `node_moved` and then prevented its Drop, we must manually
                // deallocate the heap memory here.
                let layout = std::alloc::Layout::new::<Node<K, V>>();
                std::alloc::dealloc(raw_ptr as *mut u8, layout);

                val
            };

            self.len -= 1;
            return Some(value);
        }

        // previous starts as the link we were given; iterator will start at that node
        let mut previous = start;

        // iterate nodes starting at `start`, but skip the first (we want nodes after `previous`)
        for node_ref in unsafe { start.as_ref() }.iter().skip(1) {
            // If we found the key -> remove this node
            if node_ref.key_matches(key) {
                // node_ptr is NonNull<Node<K,V>> pointing to the node to remove
                let node_ptr: Link<K, V> = node_ref.into();

                // 1) Link previous -> node.next (unlink)
                unsafe {
                    previous.as_mut().next = node_ref.next;
                }

                // 2) Now reclaim ownership of the node's allocation and extract the value.
                //
                // We used ptr::read to move the Node out of heap memory into a stack value,
                // then wrap it in ManuallyDrop so Drop does not run for that stack copy.
                // Finally we extract the value V using ptr::read and deallocate the original
                // heap allocation with the global allocator. This avoids double-drop.
                let value = unsafe {
                    let raw_ptr = node_ptr.as_ptr();

                    // Move the Node out of the allocation (bitwise move, does not call drop).
                    let node_moved: Node<K, V> = std::ptr::read(raw_ptr);
                    let mut node_moved = std::mem::ManuallyDrop::new(node_moved);

                    // Extract the value (move out) from the Item inside the node.
                    // Using ptr::read to avoid running Drop for the moved-out field later.
                    let val = std::ptr::read(&mut node_moved.item.value);

                    // We've already unlinked the node from the list; the heap allocation
                    // still needs to be deallocated. Because we used std::ptr::read to
                    // create `node_moved` and then prevented its Drop, we must manually
                    // deallocate the heap memory here.
                    let layout = std::alloc::Layout::new::<Node<K, V>>();
                    std::alloc::dealloc(raw_ptr as *mut u8, layout);

                    val
                };

                self.len -= 1;
                return Some(value);
            }

            // early exit: if keys are ordered and we've passed the key, stop searching
            if node_ref.key().borrow() > key {
                break;
            }

            // advance previous to this node
            previous = node_ref.into();
        }

        None
    }
}

impl<K, V> Drop for SkipListMap<K, V> {
    fn drop(&mut self) {
        let mut current = self.data_list;
        while let Some(mut cur) = current {
            current = unsafe { cur.as_ref().next() };
            unsafe { std::ptr::drop_in_place(cur.as_mut()) };
        }
    }
}

// TODO: I think this is unnecessary since we can just use Node iter
/// Immutabale iterator for `LinkedList`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insertion_test() {
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
    fn basic_deletion_test() {
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
    fn fast_lane_test() {
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

        // TODO: remove does not work if there is only one element in the lane
    }
}
