use roaring::RoaringBitmap;
use std::{
    cmp::max,
    ops::{BitAnd, BitOr, Sub},
    slice,
    sync::{Arc, OnceLock},
};

const SEGMENT_SIZE: usize = u32::MAX as usize + 1;

#[derive(Clone, Debug)]
enum SetInner {
    Empty,

    /// when the index size is only between 0 to u32::MAX.
    Small(Arc<RoaringBitmap>),

    /// when the index is larger that u32::MAX.
    Large(Arc<[Arc<RoaringBitmap>]>),
}

impl SetInner {
    fn from_sorted_dedup_vec(vec: Vec<usize>) -> Self {
        if vec.is_empty() {
            return Self::Empty;
        }

        // If all values are within u32 range, we can use Small
        if *vec.last().unwrap() < SEGMENT_SIZE {
            let bitmap = vec.into_iter().map(|x| x as u32).collect();
            return Self::Small(Arc::new(bitmap));
        }

        // Otherwise, build segmented bitmaps
        let mut segments = Vec::new();

        for value in vec {
            let seg_index = value / SEGMENT_SIZE;
            let seg_offset = (value % SEGMENT_SIZE) as u32;

            if seg_index >= segments.len() {
                segments.resize_with(seg_index + 1, RoaringBitmap::new);
            }

            unsafe { segments.get_unchecked_mut(seg_index) }.insert(seg_offset);
        }

        // Wrap each segment in an Arc
        let arc_segments = segments.into_iter().map(Arc::new).collect();

        Self::from_roaring_vec(arc_segments)
    }

    fn from_roaring_diff(a: &Arc<RoaringBitmap>, c: RoaringBitmap) -> Self {
        if **a == c {
            Self::Small(a.clone())
        } else if c.is_empty() {
            Self::Empty
        } else {
            Self::Small(Arc::new(c))
        }
    }

    fn from_roaring_unisect(
        a: &Arc<RoaringBitmap>,
        b: &Arc<RoaringBitmap>,
        c: RoaringBitmap,
    ) -> Self {
        if **a == c {
            Self::Small(a.clone())
        } else if **b == c {
            Self::Small(b.clone())
        } else if c.is_empty() {
            Self::Empty
        } else {
            Self::Small(Arc::new(c))
        }
    }

    fn from_roaring_vec(mut vec: Vec<Arc<RoaringBitmap>>) -> Self {
        if let Some(index) = vec.iter().rposition(|set| !set.is_empty()) {
            vec.truncate(index + 1);
        }

        if vec.len() == 1 {
            Self::Small(unsafe { vec.pop().unwrap_unchecked() })
        } else if vec.is_empty() {
            Self::Empty
        } else {
            Self::Large(vec.into())
        }
    }

    fn contains(&self, index: usize) -> bool {
        match self {
            Self::Empty => false,
            Self::Small(a) => index < SEGMENT_SIZE && a.contains(index as u32),
            Self::Large(segments) => {
                let segment_index = index / SEGMENT_SIZE;
                let inner_index = (index % SEGMENT_SIZE) as u32;

                match segments.get(segment_index) {
                    Some(set) => set.contains(inner_index),
                    None => false,
                }
            }
        }
    }

    fn difference(&self, b: &Self) -> Self {
        match (self, b) {
            (Self::Empty, _) => Self::Empty,
            (a, Self::Empty) => a.clone(),
            (Self::Small(a), Self::Small(b)) => Self::from_roaring_diff(a, &**a - &**b),
            (Self::Small(a), Self::Large(segments)) => {
                // Assume no empty segments.
                let b = unsafe { segments.get_unchecked(0) };

                Self::from_roaring_diff(a, &**a - &**b)
            }
            (Self::Large(segments), Self::Small(b)) => {
                let c = unsafe { segments.get_unchecked(0) };
                let mut vec = Vec::with_capacity(segments.len());

                vec.push(unisect_arc(c, b, &**c - &**b));

                if segments.len() > 1 {
                    vec.extend_from_slice(&segments[1..]);
                }

                SetInner::from_roaring_vec(vec)
            }
            (Self::Large(a_segments), Self::Large(b_segments)) => {
                let vec = a_segments
                    .iter()
                    .zip(b_segments.iter())
                    .map(|(a, b)| {
                        let c = &**a - &**b;

                        if **a == c {
                            a.clone()
                        } else if c.is_empty() {
                            empty_roaring_arc().clone()
                        } else {
                            Arc::new(c)
                        }
                    })
                    .collect();

                Self::from_roaring_vec(vec)
            }
        }
    }

    fn intersection(&self, b: &Self) -> Self {
        match (self, b) {
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::Small(a), Self::Small(b)) => Self::from_roaring_unisect(a, b, &**a & &**b),
            (Self::Small(a), Self::Large(segments)) | (Self::Large(segments), Self::Small(a)) => {
                // Assumes bv is not empty if it's Large
                let b = unsafe { segments.get_unchecked(0) };

                Self::from_roaring_unisect(a, b, &**a & &**b)
            }
            (Self::Large(a_segments), Self::Large(b_segments)) => {
                let vec = a_segments
                    .iter()
                    .zip(b_segments.iter())
                    .map(|(a, b)| unisect_arc(a, b, &**a & &**b))
                    .collect();

                Self::from_roaring_vec(vec)
            }
        }
    }

    fn is_subset(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, _) => true,
            (_, Self::Empty) => false,
            (Self::Small(a), Self::Small(b)) => a.is_subset(b),
            (Self::Small(a), Self::Large(b_segments)) => {
                // Assume b_segments is not empty if it's Large
                a.is_subset(&**unsafe { b_segments.get_unchecked(0) })
            }
            (Self::Large(_), Self::Small(_)) => false,
            (Self::Large(a_segments), Self::Large(b_segments)) => {
                if a_segments.len() > b_segments.len() {
                    return false;
                }
                a_segments
                    .iter()
                    .zip(b_segments.iter())
                    .all(|(a, b)| a.is_subset(b))
            }
        }
    }

    fn iter(&self) -> IterInner {
        match self {
            Self::Empty => IterInner::Empty,
            Self::Small(a) => IterInner::Small(a.iter()),
            Self::Large(segments) => {
                let mut iter = segments.iter();

                IterInner::Large {
                    // Assume Large is never empty
                    roar: unsafe { iter.next().unwrap_unchecked() }.iter(),
                    segments: iter,
                    base: 0,
                }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Small(set) => set.len() as usize,
            Self::Large(segments) => segments.iter().map(|set| set.len() as usize).sum(),
        }
    }

    fn union(&self, b: &Self) -> Self {
        match (self, b) {
            (Self::Empty, a) | (a, Self::Empty) => a.clone(),
            (Self::Small(a), Self::Small(b)) => Self::from_roaring_unisect(a, b, &**a | &**b),
            (Self::Small(a), Self::Large(segments)) | (Self::Large(segments), Self::Small(a)) => {
                let mut vec = Vec::with_capacity(segments.len());

                // Assumes segs is not empty if it's Large
                let b = unsafe { segments.get_unchecked(0) };

                vec.push(unisect_arc(a, b, &**a | &**b));

                if segments.len() > 1 {
                    vec.extend_from_slice(&segments[1..]);
                }

                // Assume no empty segments.
                Self::Large(vec.into())
            }
            (Self::Large(a_segments), Self::Large(b_segments)) => {
                let mut vec = Vec::with_capacity(max(a_segments.len(), b_segments.len()));
                let mut a_iter = a_segments.iter();
                let mut b_iter = b_segments.iter();

                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (Some(a), Some(b)) => vec.push(unisect_arc(a, b, &**a | &**b)),
                        (None, Some(bv)) => {
                            vec.push(bv.clone());
                            vec.extend(b_iter.cloned());
                            break;
                        }
                        (Some(av), None) => {
                            vec.push(av.clone());
                            vec.extend(a_iter.cloned());
                            break;
                        }
                        (None, None) => break,
                    }
                }

                // Assume no empty segments.
                Self::Large(vec.into())
            }
        }
    }
}

impl Eq for SetInner {}

impl PartialEq for SetInner {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, Self::Empty) => true,
            (Self::Small(a), Self::Small(b)) => a == b,
            (Self::Large(a_segments), Self::Large(b_segments)) => a_segments == b_segments,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SrbSet(SetInner);

impl SrbSet {
    #[inline]
    pub const fn new() -> Self {
        Self(SetInner::Empty)
    }

    /// Creates a SrbSet based on an already sorted and unique indices vec.
    ///
    /// # Safety
    /// User of this function must ensure that the vec is sorted and deduplicated.
    #[inline]
    pub unsafe fn from_sorted_dedup_vec_unchecked(vec: Vec<usize>) -> Self {
        debug_assert!(is_sorted_order(&vec), "vec not in sorted order");
        Self(SetInner::from_sorted_dedup_vec(vec))
    }

    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        self.0.contains(index)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self.0, SetInner::Empty)
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    #[inline]
    pub fn iter(&self) -> Iter {
        Iter(self.0.iter())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for SrbSet {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BitAnd for SrbSet {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0.intersection(&rhs.0))
    }
}

impl BitAnd<&SrbSet> for SrbSet {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: &Self) -> Self::Output {
        Self(self.0.intersection(&rhs.0))
    }
}

impl BitAnd for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        SrbSet(self.0.intersection(&rhs.0))
    }
}

impl BitAnd<SrbSet> for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn bitand(self, rhs: SrbSet) -> Self::Output {
        SrbSet(self.0.intersection(&rhs.0))
    }
}

impl BitOr for SrbSet {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0.union(&rhs.0))
    }
}

impl BitOr<&SrbSet> for SrbSet {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: &Self) -> Self::Output {
        Self(self.0.union(&rhs.0))
    }
}

impl BitOr for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        SrbSet(self.0.union(&rhs.0))
    }
}

impl BitOr<SrbSet> for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn bitor(self, rhs: SrbSet) -> Self::Output {
        SrbSet(self.0.union(&rhs.0))
    }
}

impl Sub for SrbSet {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(SetInner::difference(&self.0, &rhs.0))
    }
}

impl Sub<&SrbSet> for SrbSet {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &SrbSet) -> Self::Output {
        Self(self.0.difference(&rhs.0))
    }
}

impl Sub for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        SrbSet(self.0.difference(&rhs.0))
    }
}

impl Sub<SrbSet> for &SrbSet {
    type Output = SrbSet;

    #[inline]
    fn sub(self, rhs: SrbSet) -> Self::Output {
        SrbSet(self.0.difference(&rhs.0))
    }
}

impl FromIterator<usize> for SrbSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut vec = iter.into_iter().collect::<Vec<_>>();

        vec.sort_unstable();
        vec.dedup();

        Self(SetInner::from_sorted_dedup_vec(vec))
    }
}

pub struct Iter<'a>(IterInner<'a>);

impl Iterator for Iter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

enum IterInner<'a> {
    Empty,
    Small(roaring::bitmap::Iter<'a>),
    Large {
        base: usize,
        roar: roaring::bitmap::Iter<'a>,
        segments: slice::Iter<'a, Arc<RoaringBitmap>>,
    },
}

impl IterInner<'_> {
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Empty => None,
            Self::Small(a) => a.next().map(|v| v as usize),
            Self::Large {
                roar,
                segments,
                base,
            } => loop {
                if let Some(v) = roar.next() {
                    return Some(v as usize + *base);
                }

                match segments.next() {
                    Some(r) => {
                        *base += SEGMENT_SIZE;
                        *roar = r.iter()
                    }
                    None => {
                        *self = Self::Empty;
                        return None;
                    }
                }
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Empty => (0, Some(0)),
            Self::Small(a) => a.size_hint(),
            Self::Large { roar, .. } => (roar.size_hint().0, None),
        }
    }
}

fn empty_roaring_arc() -> &'static Arc<RoaringBitmap> {
    static CELL: OnceLock<Arc<RoaringBitmap>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(RoaringBitmap::new()))
}

fn is_sorted_order(s: &[usize]) -> bool {
    let mut iterb = s.iter();
    iterb.next();

    s.iter().zip(iterb).all(|(a, b)| a < b)
}

fn unisect_arc(
    a: &Arc<RoaringBitmap>,
    b: &Arc<RoaringBitmap>,
    c: RoaringBitmap,
) -> Arc<RoaringBitmap> {
    if **a == c {
        a.clone()
    } else if **b == c {
        b.clone()
    } else if c.is_empty() {
        empty_roaring_arc().clone()
    } else {
        Arc::new(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_set() {
        let set = SrbSet::new();
        assert!(!set.contains(0));
        assert_eq!(set.iter().count(), 0);
    }

    #[test]
    fn test_single_segment() {
        let indices = vec![1, 2, 3, 100, 1000];
        let set: SrbSet = indices.clone().into_iter().collect();
        for &i in &indices {
            assert!(set.contains(i));
        }
        assert_eq!(set.iter().collect::<Vec<_>>(), indices);
    }

    #[test]
    fn test_multi_segment() {
        let seg1 = 0;
        let seg2 = SEGMENT_SIZE;
        let seg3 = SEGMENT_SIZE * 2 + 123;

        let indices = vec![seg1, seg2, seg3];
        let set: SrbSet = indices.clone().into_iter().collect();

        for &i in &indices {
            assert!(set.contains(i));
        }

        assert_eq!(set.iter().collect::<Vec<_>>(), indices);
    }

    #[test]
    fn test_union() {
        let a: SrbSet = vec![1, 2, 3].into_iter().collect();
        let b: SrbSet = vec![3, 4, 5].into_iter().collect();
        let result = &a | &b;
        let expected: Vec<_> = vec![1, 2, 3, 4, 5];
        assert_eq!(result.iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_intersection() {
        let a: SrbSet = vec![1, 2, 3].into_iter().collect();
        let b: SrbSet = vec![3, 4, 5].into_iter().collect();
        let result = &a & &b;
        assert_eq!(result.iter().collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn test_difference() {
        let a: SrbSet = vec![1, 2, 3, 4].into_iter().collect();
        let b: SrbSet = vec![3, 4, 5].into_iter().collect();
        let result = &a - &b;
        assert_eq!(result.iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn test_large_segment_operations() {
        let base = SEGMENT_SIZE;
        let a: SrbSet = vec![1, base + 1].into_iter().collect();
        let b: SrbSet = vec![base + 1].into_iter().collect();
        let union = &a | &b;
        let intersection = &a & &b;
        let difference = &a - &b;

        assert!(union.contains(1));
        assert!(union.contains(base + 1));

        assert!(!intersection.contains(1));
        assert!(intersection.contains(base + 1));

        assert!(difference.contains(1));
        assert!(!difference.contains(base + 1));
    }

    #[test]
    fn test_deduplication_and_ordering() {
        let input = vec![5, 1, 3, 1, 5, 3];
        let set: SrbSet = input.into_iter().collect();
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![1, 3, 5]);
    }

    #[test]
    fn test_is_subset_empty() {
        let empty_set = SrbSet::new();
        let small_set: SrbSet = vec![1, 2].into_iter().collect();
        let large_set: SrbSet = vec![1, SEGMENT_SIZE + 1].into_iter().collect();

        assert!(empty_set.is_subset(&empty_set)); // Empty is subset of empty
        assert!(empty_set.is_subset(&small_set)); // Empty is subset of small
        assert!(empty_set.is_subset(&large_set)); // Empty is subset of large
    }

    #[test]
    fn test_is_subset_small_to_small() {
        let s1: SrbSet = vec![1, 2].into_iter().collect();
        let s2: SrbSet = vec![1, 2, 3].into_iter().collect();
        let s3: SrbSet = vec![4, 5].into_iter().collect();

        assert!(s1.is_subset(&s2));
        assert!(!s2.is_subset(&s1));
        assert!(s1.is_subset(&s1));
        assert!(!s1.is_subset(&s3));
    }

    #[test]
    fn test_is_subset_small_to_large() {
        let s_small: SrbSet = vec![10, 20].into_iter().collect();
        let l_large: SrbSet = vec![5, 10, 20, 30, SEGMENT_SIZE + 1].into_iter().collect();
        let l_large_no_first_seg: SrbSet = vec![SEGMENT_SIZE + 1].into_iter().collect();

        assert!(s_small.is_subset(&l_large));
        assert!(!s_small.is_subset(&l_large_no_first_seg)); // Small elements can't be in higher segments
        assert!(!l_large.is_subset(&s_small)); // Large cannot be subset of small (generally)
    }

    #[test]
    fn test_is_subset_large_to_small() {
        let l_large: SrbSet = vec![10, SEGMENT_SIZE + 1].into_iter().collect();
        let s_small: SrbSet = vec![5, 10, 15].into_iter().collect();
        let l_single_segment: SrbSet = vec![5, 10].into_iter().collect(); // This will internally be Small
        let s_superset: SrbSet = vec![0, 1, 2, 3, 4, 5, 10].into_iter().collect();

        assert!(!l_large.is_subset(&s_small)); // Multi-segment large cannot be subset of small
        assert!(l_single_segment.is_subset(&s_superset)); // Single-segment large (acting as Small) is subset of Small
        assert!(!s_small.is_subset(&l_large)); // Small is not subset of large if elements mismatch
    }

    #[test]
    fn test_is_subset_large_to_large() {
        let l1: SrbSet = vec![1, SEGMENT_SIZE + 1].into_iter().collect();
        let l2: SrbSet = vec![1, 2, SEGMENT_SIZE + 1, SEGMENT_SIZE + 2]
            .into_iter()
            .collect();
        let l3: SrbSet = vec![1, SEGMENT_SIZE * 2 + 1].into_iter().collect(); // l3 has a segment l2 doesn't have at that index

        assert!(l1.is_subset(&l2));
        assert!(!l2.is_subset(&l1));
        assert!(l1.is_subset(&l1));
        assert!(!l1.is_subset(&l3)); // l1[1] is not subset of l3[1] because l3[1] is empty
        assert!(!l3.is_subset(&l1)); // l3[1] is not subset of l1[1] because l1[1] is empty

        let l_larger_len: SrbSet = vec![1, SEGMENT_SIZE + 1, SEGMENT_SIZE * 2 + 1]
            .into_iter()
            .collect();
        assert!(!l_larger_len.is_subset(&l2)); // l_larger_len has more segments than l2
    }

    #[test]
    fn test_is_subset_complex_mixed_difference_case() {
        let large_set_a: SrbSet = vec![1, 2, SEGMENT_SIZE + 1, SEGMENT_SIZE * 2 + 5]
            .into_iter()
            .collect();
        let small_set_b: SrbSet = vec![2].into_iter().collect();

        let result_diff = &large_set_a - &small_set_b;
        let expected_diff: SrbSet = vec![1, SEGMENT_SIZE + 1, SEGMENT_SIZE * 2 + 5]
            .into_iter()
            .collect();

        assert_eq!(result_diff, expected_diff);
        assert!(result_diff.is_subset(&large_set_a));
        assert!(!large_set_a.is_subset(&result_diff)); // large_set_a has '2', result_diff does not
    }

    // Test Case 1: Empty Slice
    #[test]
    fn test_empty_slice() {
        let s: &[usize] = &[];
        assert_eq!(
            is_sorted_order(s),
            true,
            "Empty slice should be considered sorted."
        );
    }

    // Test Case 2: Single Element Slice
    #[test]
    fn test_single_element_slice() {
        let s = &[5];
        assert_eq!(
            is_sorted_order(s),
            true,
            "Single-element slice should be considered sorted."
        );
    }

    // Test Case 3: Strictly Sorted Slice
    #[test]
    fn test_strictly_sorted_slice() {
        let s = &[1, 2, 3, 4, 5];
        assert_eq!(
            is_sorted_order(s),
            true,
            "Strictly sorted slice should return true."
        );
    }

    // Test Case 4: Slice with Duplicate Elements
    #[test]
    fn test_duplicate_elements() {
        let s = &[1, 2, 2, 3, 4];
        assert_eq!(
            is_sorted_order(s),
            false,
            "Slice with duplicates should return false."
        );
    }

    // Test Case 5: Slice in Descending Order
    #[test]
    fn test_descending_order() {
        let s = &[5, 4, 3, 2, 1];
        assert_eq!(
            is_sorted_order(s),
            false,
            "Descending slice should return false."
        );
    }

    // Test Case 6: Slice with Mixed Unsorted Elements
    #[test]
    fn test_mixed_unsorted_elements() {
        let s = &[1, 3, 2, 4];
        assert_eq!(
            is_sorted_order(s),
            false,
            "Mixed unsorted slice should return false."
        );
    }

    // Test Case 7: Two Elements, Sorted
    #[test]
    fn test_two_elements_sorted() {
        let s = &[10, 20];
        assert_eq!(
            is_sorted_order(s),
            true,
            "Two sorted elements should return true."
        );
    }

    // Test Case 8: Two Elements, Unsorted
    #[test]
    fn test_two_elements_unsorted() {
        let s = &[20, 10];
        assert_eq!(
            is_sorted_order(s),
            false,
            "Two unsorted elements should return false."
        );
    }

    // Test Case 9: All Elements Are The Same
    #[test]
    fn test_all_same_elements() {
        let s = &[7, 7, 7];
        assert_eq!(
            is_sorted_order(s),
            false,
            "Slice with all identical elements should return false."
        );
    }
}
