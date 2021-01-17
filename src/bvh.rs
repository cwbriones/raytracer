use std::sync::Arc;

use rand::Rng;

use crate::trace::{
    Bounded,
    Hit,
    Hittable,
    Ray,
    AABB,
};
use crate::util::NonNan;

/// The cost of computing an intersection given a ray.
const INTERSECT_COST: f64 = 1.0;

/// The cost of traversing from a parent to child node in the tree.
const TRAVERSAL_COST: f64 = 2.0;

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
#[derive(Clone)]
pub struct BVH<S> {
    root: BVHNode<S>,
}

impl<S> BVH<S>
where
    S: Hittable + Bounded + Clone,
{
    pub fn new(surfaces: &mut [S]) -> Self {
        BVH {
            root: BVHNode::new_with_strategy(surfaces, &mut SAHConstructionStrategy),
        }
    }
}

impl<S> Hittable for BVH<S>
where
    S: Hittable + Bounded,
{
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        self.root.hit(ray, t_min, t_max)
    }
}

/// A strategy for constructing a BVH.
trait ConstructionStrategy<S> {
    /// Choose the axis used to partition this level of the tree.
    fn choose_axis(&mut self, surfaces: &[S]) -> usize;

    /// Choose the partitioning index. This is the index of the first surface
    /// that will be placed in the right child.
    ///
    /// If `None` is returned, then a leaf node containing these surfaces
    /// will be constructed.
    fn split_at(&mut self, surfaces: &[S]) -> Option<usize>;
}

/// Construct the tree randomly.
struct RandomConstructionStrategy<R>(R);

impl<S, R> ConstructionStrategy<S> for RandomConstructionStrategy<R>
where
    R: Rng,
{
    fn choose_axis(&mut self, _surfaces: &[S]) -> usize {
        self.0.gen_range(0..3)
    }

    fn split_at(&mut self, surfaces: &[S]) -> Option<usize> {
        Some(surfaces.len() / 2)
    }
}

/// Construct the tree using the surface area heuristic.
struct SAHConstructionStrategy;

impl<S> ConstructionStrategy<S> for SAHConstructionStrategy
where
    S: Bounded,
{
    fn choose_axis(&mut self, surfaces: &[S]) -> usize {
        // Choose the axis that has the widest span of centroids.
        //
        // This is the distance between the rightmost and leftmost
        // bounding boxes on a given axis.
        (0usize..3)
            .max_by_key(|i| {
                let mut min = f64::INFINITY;
                let mut max = 0.0;
                for surface in surfaces.iter() {
                    let p = surface.bounding_box().centroid().get(*i);
                    if p < min {
                        min = p;
                    }
                    if p > max {
                        max = p;
                    }
                }
                NonNan::new(max - min).unwrap()
            })
            .unwrap()
    }

    fn split_at(&mut self, surfaces: &[S]) -> Option<usize> {
        let mut root_bound = surfaces[0].bounding_box();
        for surface in &surfaces[1..] {
            root_bound = root_bound.merge(&surface.bounding_box());
        }

        let (best_split, best_cost) = (1..surfaces.len())
            .map(|split_idx| {
                // Left box
                let mut left = surfaces[0].bounding_box();
                for surface in &surfaces[1..split_idx] {
                    left = left.merge(&surface.bounding_box());
                }
                // Right box
                let mut right = surfaces[split_idx].bounding_box();
                for surface in &surfaces[split_idx..] {
                    right = right.merge(&surface.bounding_box());
                }
                let split_cost = TRAVERSAL_COST
                    + left.surface_area() * split_idx as f64 * INTERSECT_COST
                    + right.surface_area() * (surfaces.len() - split_idx) as f64 * INTERSECT_COST;
                (split_idx, split_cost)
            })
            .min_by_key(|(_, cost)| NonNan::new(*cost).unwrap())
            .unwrap();

        if best_cost > (root_bound.surface_area() * surfaces.len() as f64 * INTERSECT_COST) {
            // It's cheaper to make a leaf at this level than it is to split.
            None
        } else {
            Some(best_split)
        }
    }
}

#[derive(Clone)]
enum BVHNode<S> {
    Inner(BVHInnerNode<S>),
    Leaf(BVHLeafNode<S>),
}

impl<S> BVHNode<S>
where
    S: Hittable + Bounded + Clone,
{
    fn new_with_strategy<T>(surfaces: &mut [S], strategy: &mut T) -> Self
    where
        T: ConstructionStrategy<S>,
    {
        // Choose the axis
        let axis = strategy.choose_axis(surfaces);
        let comparator = |a: &S, b: &S| {
            let bba = a.bounding_box();
            let bbb = b.bounding_box();
            bba.min()
                .get(axis)
                .partial_cmp(&bbb.min().get(axis))
                .unwrap()
        };
        let min_split_len = 4;
        if surfaces.len() <= min_split_len {
            return BVHNode::Leaf(BVHLeafNode::new(surfaces.to_vec()));
        }
        // Subdivide.
        surfaces.sort_by(comparator);
        if let Some(idx) = strategy.split_at(surfaces) {
            let left = Arc::new(BVHNode::new_with_strategy(&mut surfaces[..idx], strategy));
            let right = Arc::new(BVHNode::new_with_strategy(&mut surfaces[idx..], strategy));
            let box_left = left.bounding_box();
            let box_right = right.bounding_box();
            let bound = box_left.merge(&box_right);
            return BVHNode::Inner(BVHInnerNode {
                left: Some(left),
                right: Some(right),
                bound,
            });
        }
        BVHNode::Leaf(BVHLeafNode::new(surfaces.to_vec()))
    }
}

impl<S> BVHNode<S>
where
    S: Hittable + Bounded,
{
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        match *self {
            Self::Inner(ref inner) => inner.hit(ray, t_min, t_max),
            Self::Leaf(ref leaf) => leaf.hit(ray, t_min, t_max),
        }
    }

    fn bounding_box(&self) -> &AABB {
        match *self {
            Self::Inner(ref inner) => inner.bounding_box(),
            Self::Leaf(ref leaf) => leaf.bounding_box(),
        }
    }
}

#[derive(Clone)]
struct BVHInnerNode<S> {
    left: Option<Arc<BVHNode<S>>>,
    right: Option<Arc<BVHNode<S>>>,
    bound: AABB,
}

impl<S> BVHInnerNode<S>
where
    S: Hittable + Bounded,
{
    #[inline(always)]
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        if !self.bound.hit(ray, t_min, t_max) {
            return None;
        }
        let hit_left = self.left.as_ref().and_then(|n| n.hit(ray, t_min, t_max));
        let hit_right = self.right.as_ref().and_then(|n| n.hit(ray, t_min, t_max));
        match (hit_left, hit_right) {
            (None, None) => None,
            (hit @ Some(_), None) => hit,
            (None, hit @ Some(_)) => hit,
            (Some(a), Some(b)) if a.t < b.t => Some(a),
            (_, Some(b)) => Some(b),
        }
    }

    fn bounding_box(&self) -> &AABB {
        &self.bound
    }
}

#[derive(Clone)]
struct BVHLeafNode<S> {
    surfaces: Arc<[S]>,
    bound: AABB,
}

impl<S> BVHLeafNode<S>
where
    S: Hittable + Bounded,
{
    fn new(surfaces: Vec<S>) -> Self {
        let mut bound = surfaces[0].bounding_box();
        for obj in &surfaces[1..] {
            bound = bound.merge(&obj.bounding_box());
        }
        Self {
            surfaces: surfaces.into(),
            bound,
        }
    }

    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut closest = t_max;
        let mut closest_hit = None;
        for o in &self.surfaces[..] {
            if let Some(hit) = o.hit(ray, t_min, closest) {
                if hit.t < closest {
                    closest = hit.t;
                    closest_hit = Some(hit);
                }
            }
        }
        closest_hit
    }

    fn bounding_box(&self) -> &AABB {
        &self.bound
    }
}
