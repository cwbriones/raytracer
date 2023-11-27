use std::sync::Arc;

use super::Surface;
use crate::geom::Point3;
use crate::trace::{
    Aabb,
    Bounded,
    Hit,
    Hittable,
    Interval,
    Ray,
};

/// The cost of computing an intersection given a ray.
const INTERSECT_COST: f64 = 1.0;

/// The cost of traversing from a parent to child node in the tree.
const TRAVERSAL_COST: f64 = 2.0;

// The maximum number of surfaces within a single leaf when built automatically.
//
// It is still possible to create leaves with more elements manually.
const MAX_LEAF_SIZE: usize = 8;

/// A bounded volume hiearchy (BVH).
///
/// A BVH is a tracing acceleration structure which seeks to minimize
/// the number of surfaces we need to check for intersection. It does this
/// by partitioning a scene into a set of nested bounding boxes (hence "hierarchy").
/// If a ray does not intersect a box, then it is guaranteed not to intersect any
/// of the surfaces contained within that box.
///
/// This implementation in particular uses what is known as the surface area heuristic (SAH)
/// to assist in construction. This heuristic operates under the observation that the probability
/// of intersection with a box is proportional to its surface area. So along with the cost of any
/// subsequent intersections that could occur, we can determine whether or not to split at a given
/// level by comparing costs of all possible splits in addition to the option of not splitting at
/// all.
#[derive(Debug, Clone)]
pub struct Bvh {
    root: BVHNode,
}

impl Bvh {
    /// Construct a BVH using the Surface-Area-Heuristic.
    #[allow(unused)]
    pub fn new(surfaces: Vec<Surface>) -> Self {
        Bvh {
            root: BVHNode::new_with_strategy(surfaces, &mut SAHSplitStrategy),
        }
    }

    /// Construct a BVH by dividing each level evenly each time.
    #[allow(unused)]
    pub fn new_naive(surfaces: Vec<Surface>) -> Self {
        Bvh {
            root: BVHNode::new_with_strategy(surfaces, &mut EqualSplitStrategy),
        }
    }

    /// Construct a BVH that consists of a single leaf node.
    #[allow(unused)]
    pub fn new_leaf(surfaces: Vec<Surface>) -> Self {
        let surfaces = surfaces.to_vec();
        Bvh {
            root: BVHNode::Leaf(BVHLeafNode::new(surfaces)),
        }
    }

    pub fn builder() -> BvhBuilder {
        BvhBuilder {
            surfaces: Vec::new(),
        }
    }
}

/// BvhBuilder is a convenience struct that handles conversion to surfaces when
/// you do not already have a Vec<Surface> allocated.
pub struct BvhBuilder {
    surfaces: Vec<Surface>,
}

impl BvhBuilder {
    pub fn add<S: Into<Surface>>(&mut self, surface: S) {
        self.surfaces.push(surface.into());
    }

    pub fn build(self) -> Bvh {
        Bvh::new(self.surfaces)
    }

    pub fn build_leaf(self) -> Bvh {
        Bvh::new_leaf(self.surfaces)
    }
}

impl Hittable for Bvh {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        self.root.hit(ray, interval)
    }
}

impl Bounded for Bvh {
    fn bounding_box(&self) -> Aabb {
        self.root.bounding_box().clone()
    }
}

/// A strategy for constructing a BVH.
trait SplitStrategy {
    /// Choose the partitioning index. This is the index of the first surface
    /// that will be placed in the right child.
    ///
    /// If `None` is returned, then a leaf node containing these surfaces
    /// will be constructed.
    fn split_at(&mut self, surfaces: &[Surface]) -> Option<usize>;
}

/// Split each level of the tree equally into two parts.
struct EqualSplitStrategy;

impl SplitStrategy for EqualSplitStrategy {
    fn split_at(&mut self, surfaces: &[Surface]) -> Option<usize> {
        Some(surfaces.len() / 2)
    }
}

/// Split each level of the tree using the surface area heuristic.
struct SAHSplitStrategy;

impl SplitStrategy for SAHSplitStrategy {
    fn split_at(&mut self, surfaces: &[Surface]) -> Option<usize> {
        // Accumulate boxes from left to right.
        let mut merge_left = Vec::with_capacity(surfaces.len());
        merge_left.push(surfaces[0].bounding_box());
        for s in &surfaces[1..] {
            let last = merge_left.last().unwrap();
            let merged = s.bounding_box().merge(last);
            merge_left.push(merged);
        }

        // Accumulate boxes from right to left.
        let mut merge_right = Vec::with_capacity(surfaces.len());
        merge_right.push(surfaces.last().unwrap().bounding_box());
        for s in surfaces.iter().rev().skip(1) {
            let last = merge_right.last().unwrap();
            let merged = s.bounding_box().merge(last);
            merge_right.push(merged);
        }
        let root_bound = merge_left.last().unwrap();

        let (best_split, best_cost) = (1..surfaces.len() - 1)
            .map(|split_idx| {
                // Left box
                let left = &merge_left[split_idx - 1];
                // The smallest box for right is at index 0 and indicates splitting at `len() - 2`.
                // The largest box is at index `len() - 1` and indices no splitting at all.
                let right = &merge_right[merge_right.len() - split_idx - 1];
                let split_cost = TRAVERSAL_COST
                    + left.surface_area() * split_idx as f64 * INTERSECT_COST
                    + right.surface_area() * (surfaces.len() - split_idx) as f64 * INTERSECT_COST;
                (split_idx, split_cost)
            })
            .min_by(|(_, costa), (_, costb)| costa.partial_cmp(costb).expect("not NaN"))
            .unwrap();

        if best_cost >= (root_bound.surface_area() * surfaces.len() as f64 * INTERSECT_COST) {
            // It's cheaper to make a leaf at this level than it is to split.
            None
        } else {
            Some(best_split)
        }
    }
}

#[derive(Debug, Clone)]
enum BVHNode {
    Inner(BVHInnerNode),
    Leaf(BVHLeafNode),
}

impl BVHNode {
    fn new_with_strategy<T>(mut surfaces: Vec<Surface>, strategy: &mut T) -> Self
    where
        T: SplitStrategy,
    {
        if surfaces.len() <= MAX_LEAF_SIZE {
            return BVHNode::Leaf(BVHLeafNode::new(surfaces.to_vec()));
        }
        // Choose the axis that has the widest span of centroids.
        //
        // This is the distance between the rightmost and leftmost
        // bounding boxes on a given axis.
        let mut mins = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut maxs = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for surface in surfaces.iter() {
            let centroid = surface.bounding_box().centroid();
            mins = mins.min_pointwise(&centroid);
            maxs = maxs.max_pointwise(&centroid);
        }
        let span = maxs - mins;
        let mut axis = 0;
        if span.get(axis) < span.get(1) {
            axis = 1
        }
        if span.get(axis) < span.get(2) {
            axis = 2
        }
        let comparator = |a: &Surface, b: &Surface| {
            let bba = a.bounding_box();
            let bbb = b.bounding_box();
            bba.min()
                .get(axis)
                .partial_cmp(&bbb.min().get(axis))
                .unwrap()
        };
        // Subdivide along this axis.
        surfaces.sort_by(comparator);
        if let Some(idx) = strategy.split_at(&surfaces) {
            let right = surfaces.split_off(idx);

            let left = Arc::new(BVHNode::new_with_strategy(surfaces, strategy));
            let right = Arc::new(BVHNode::new_with_strategy(right, strategy));
            let box_left = left.bounding_box();
            let box_right = right.bounding_box();
            let bound = box_left.merge(box_right);
            return BVHNode::Inner(BVHInnerNode {
                left: Some(left),
                right: Some(right),
                bound,
            });
        }
        BVHNode::Leaf(BVHLeafNode::new(surfaces.to_vec()))
    }
}

impl BVHNode {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        match *self {
            Self::Inner(ref inner) => inner.hit(ray, interval),
            Self::Leaf(ref leaf) => leaf.hit(ray, interval),
        }
    }

    fn bounding_box(&self) -> &Aabb {
        match *self {
            Self::Inner(ref inner) => inner.bounding_box(),
            Self::Leaf(ref leaf) => &leaf.bound,
        }
    }
}

#[derive(Debug, Clone)]
struct BVHInnerNode {
    left: Option<Arc<BVHNode>>,
    right: Option<Arc<BVHNode>>,
    bound: Aabb,
}

impl BVHInnerNode {
    #[inline(always)]
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        if !self.bound.hit(ray, interval) {
            return None;
        }
        let hit_left = self.left.as_ref().and_then(|n| n.hit(ray, interval));
        let hit_right = self.right.as_ref().and_then(|n| n.hit(ray, interval));
        match (hit_left, hit_right) {
            (None, None) => None,
            (hit @ Some(_), None) => hit,
            (None, hit @ Some(_)) => hit,
            (Some(a), Some(b)) if a.t < b.t => Some(a),
            (_, Some(b)) => Some(b),
        }
    }

    fn bounding_box(&self) -> &Aabb {
        &self.bound
    }
}

#[derive(Debug, Clone)]
struct BVHLeafNode {
    surfaces: Arc<[Surface]>,
    bound: Aabb,
}

impl BVHLeafNode {
    fn new(surfaces: Vec<Surface>) -> Self {
        let mut bound = surfaces[0].bounding_box();
        for obj in &surfaces[1..] {
            bound = bound.merge(&obj.bounding_box());
        }
        Self {
            surfaces: surfaces.into(),
            bound,
        }
    }

    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        if !self.bound.hit(ray, interval) {
            return None;
        }
        let Interval(t_min, t_max) = interval;
        let mut closest = t_max;
        let mut closest_hit = None;
        for o in &self.surfaces[..] {
            if let Some(hit) = o.hit(ray, Interval(t_min, closest)) {
                closest = hit.t;
                closest_hit = Some(hit);
            }
        }
        closest_hit
    }
}
