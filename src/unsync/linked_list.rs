// This implementation inspired by the book https://rust-unofficial.github.io/too-many-lists/index.html

use std::{marker::PhantomData, ptr::NonNull};

/// A `Node` in a `LinkedList`.
#[derive(Debug)]
pub struct Node<T> {
    value: T,
    next: Link<T>,
}

impl<T> Node<T> {
    /// Create a new node from a value.
    pub fn new(value: T) -> Self {
        Self { value, next: None }
    }
}

impl<T> PartialEq for Node<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Eq for Node<T> where T: Eq {}

// This is just because I am tired of typing Link<T>
type Link<T> = Option<NonNull<Node<T>>>;

/// Creates a new Link from a value (not wrapped in an Option)
fn new_link<T>(value: T) -> NonNull<Node<T>> {
    unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(value)))) }
}

/// A singly non-concurrent `LinkedList`.
pub struct LinkedList<T> {
    head: Link<T>,
    len: usize,
}

impl<T> LinkedList<T> {
    /// Create a new empty `LinkedList`.
    pub fn new() -> Self {
        Self { head: None, len: 0 }
    }

    /// Adds an element to the front of the `LinkedList` making it the new head and increasing its length.
    /// Time complexity: *O*(*1*).
    pub fn push_front(&mut self, element: T) {
        // create node
        let mut node = new_link(element);

        // replacing the head and getting the old one with the new value
        let old_head = self.head.replace(node);
        // all this is save, we literally control all the memory
        unsafe { node.as_mut().next = old_head };

        // incrementing the length
        self.len += 1;
    }

    /// Removes and returns the head element from the `LinkedList` and decreases the length.
    /// Time complexity: *O*(*1*).
    pub fn pop_front(&mut self) -> Option<T> {
        // checking if there is a head
        let head = self.head?;

        // destructuring the head
        let Node { value, next } = *unsafe { Box::from_raw(head.as_ptr()) };

        // updating the head
        self.head = next;

        // A subtracting won't occur if the len is 0 since the head is None which cannot happen
        self.len = unsafe { self.len.unchecked_sub(1) };

        Some(value)
    }

    /// Returns a reference to the head of the `LinkedList` or None if it does not exist.
    /// Time complexity: *O*(*1*).
    pub fn front(&self) -> Option<&T> {
        self.head.map(|node| unsafe { &node.as_ref().value })
    }

    /// Returns a mutable reference to the head of the `LinkedList` or None if it does not exist.
    /// Time complexity: *O*(*1*).
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.head
            .map(|mut node| unsafe { &mut node.as_mut().value })
    }

    /// Splits the `LinkedList` into two seperate linked list at the specified index. The latter half of the `LinkedList` is returned from the index specified.
    /// Time complexity: *O*(*n*).
    /// Panics: index > len
    pub fn split_off(&mut self, index: usize) -> Self {
        assert!(
            index <= self.len,
            "Cannot split at index larger than the list's length"
        );

        // if at beginning just empty current list
        if index == 0 {
            let mut right = Self::default();
            std::mem::swap(self, &mut right);

            return right;
        }

        // if at the end just return an empty list
        if index == self.len {
            return Self::default();
        }

        let mut previous = self.head;
        let mut current = self.head;

        for _ in 0..index {
            // We checked the length is valid so we can do this
            previous = current;
            current = unsafe { current.unwrap_unchecked().as_ref() }.next;
        }

        // len for the returned list
        let len = unsafe { self.len.unchecked_sub(index) };

        // seperating the left list from the right
        unsafe { previous.unwrap_unchecked().as_mut() }.next = None;
        self.len = unsafe { self.len.unchecked_sub(len) };

        Self {
            head: current,
            len: len,
        }
    }

    /// Removes all elements of the `LinkedList`
    /// Time complexity: *O*(*n*).
    pub fn clear(&mut self) {
        // Since popping should take care of cleaning up the memory we can just pop until all of it is cleaned up
        while self.pop_front().is_some() {}
    }

    /// Returns true if the `LinkedList` contains the provided value
    /// Time complexity: *O*(*n*).
    pub fn contains(&self, val: &T) -> bool
    where
        T: PartialEq,
    {
        self.iter().any(|v| v == val)
    }

    /// Returns the length of the `LinkedList`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `LinkedList` is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create an iterator over immutable references
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            current: self.head,
            len: self.len(),
            _marker: PhantomData,
        }
    }

    /// Create an iterator over mutable references
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            current: self.head,
            len: self.len(),
            _marker: PhantomData,
        }
    }
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for LinkedList<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let head = match self.head {
            Some(head) => head,
            None => return Self::default(),
        };

        let head = new_link(unsafe { head.as_ref() }.value.clone());
        let mut current = head;

        for val in self.iter().skip(1) {
            let link = new_link(val.clone());
            unsafe { current.as_mut() }.next = Some(link);
            current = link;
        }

        Self {
            len: self.len(),
            head: Some(head),
        }
    }
}

impl<T> PartialEq for LinkedList<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<T> Eq for LinkedList<T> where T: Eq {}

impl<T, const N: usize> From<[T; N]> for LinkedList<T> {
    fn from(value: [T; N]) -> Self {
        Self::from_iter(value)
    }
}

impl<T> std::fmt::Debug for LinkedList<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<K> FromIterator<K> for LinkedList<K> {
    fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let val = match iter.next() {
            Some(val) => val,
            None => return Self::default(),
        };

        let head = new_link(val);
        let mut current = head;

        let mut len = 1;
        for val in iter {
            let link = new_link(val);
            unsafe { current.as_mut() }.next = Some(link);
            current = link;

            len += 1;
        }

        Self {
            len: len,
            head: Some(head),
        }
    }
}

// Got to make sure we clean up the memory
impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Immutabale iterator for `LinkedList`.
pub struct Iter<'a, T: 'a> {
    current: Link<T>,
    // technically the len isn't needed but if we have it we can provide a size_hint implementation
    len: usize,
    _marker: PhantomData<&'a Link<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            // unsafe code for the win
            let node = unsafe { self.current.unwrap_unchecked().as_ref() };
            // again this should be safe unless intrinsics get violated
            self.len = unsafe { self.len.unchecked_sub(1) };

            // simply advance forward
            self.current = node.next;

            Some(&node.value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutable iterator for `LinkedList`.
pub struct IterMut<'a, T: 'a> {
    current: Link<T>,
    // technically the len isn't needed but if we have it we can provide a size_hint implementation
    len: usize,
    _marker: PhantomData<&'a mut Link<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            // unsafe code for the win
            let node = unsafe { self.current.unwrap_unchecked().as_mut() };
            // again this should be safe unless intrinsics get violated
            self.len = unsafe { self.len.unchecked_sub(1) };

            // simply advance forward
            self.current = node.next;

            Some(&mut node.value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Owned iterator for `LinkedList`.
pub struct IntoIter<T> {
    list: LinkedList<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }
}

impl<T> IntoIterator for LinkedList<T> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { list: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_list() -> LinkedList<i32> {
        LinkedList::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    }

    #[test]
    fn push_and_iterate() {
        let mut list = create_list();

        assert!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter().eq(&list));

        list.push_front(17);
        list.push_front(20);

        assert!([20i32, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter().eq(&list));

        let mut i = 0;
        let vals = [20i32, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        while let Some(val) = list.pop_front() {
            assert_eq!(val, vals[i]);
            i += 1;
        }

        let mut list = create_list();
        let mut list: LinkedList<_> = list.iter_mut().map(|val| *val + 1).collect();

        assert!([2i32, 3, 4, 5, 6, 7, 8, 9, 10, 11].iter().eq(&list));

        list.clear();

        assert_eq!(0, list.len());
    }

    #[test]
    fn split() {
        let mut l1 = create_list();
        let copy = l1.clone();
        let mut l2 = l1.split_off(0);

        assert_eq!(l1, LinkedList::default());
        assert_eq!(l2, copy);

        let mut l3 = l2.split_off(5);

        assert!([1i32, 2, 3, 4, 5].iter().eq(&l2));
        assert!([6i32, 7, 8, 9, 10].iter().eq(&l3));

        let l4 = l3.split_off(1);

        assert!([6i32].iter().eq(&l3));
        assert!([7i32, 8, 9, 10].iter().eq(&l4));

        l3.push_front(100);
        assert!([100i32, 6].iter().eq(&l3));

        assert!(l4.contains(&9))
    }
}
