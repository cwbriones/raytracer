use std::sync::Arc;

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

#[derive(Clone)]
pub struct BVH<H> {
    root: BVHNode<H>,
}

impl<H> BVH<H>
where
    H: Hittable + Bounded + Clone,
{
    pub fn new(objects: &mut [H]) -> Self {
        BVH {
            root: BVHNode::new(objects),
        }
    }
}

impl<H> Hittable for BVH<H>
where
    H: Hittable + Bounded,
{
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        self.root.hit(ray, t_min, t_max)
    }
}

#[derive(Clone)]
enum BVHNode<H> {
    Inner(BVHInnerNode<H>),
    Leaf(BVHLeafNode<H>),
}

impl<H> BVHNode<H>
where
    H: Hittable + Bounded + Clone,
{
    fn new(objects: &mut [H]) -> Self {
        // Choose the axis
        let axis = (0usize..3)
            .max_by_key(|i| {
                // Choose the axis that has the widest span of centroids.
                //
                // This is the distance between the rightmost and leftmost
                // bounding boxes on a given axis.
                let mut min = f64::INFINITY;
                let mut max = 0.0;
                for sphere in objects.iter() {
                    let p = sphere.bounding_box().centroid().get(*i);
                    if p < min {
                        min = p;
                    }
                    if p > max {
                        max = p;
                    }
                }
                NonNan::new(max - min).unwrap()
            })
            .unwrap();
        let comparator = |a: &H, b: &H| {
            let bba = a.bounding_box();
            let bbb = b.bounding_box();
            bba.min()
                .get(axis)
                .partial_cmp(&bbb.min().get(axis))
                .unwrap()
        };
        let min_split_len = 4;
        if objects.len() <= min_split_len {
            return BVHNode::Leaf(BVHLeafNode::new(objects.to_vec()));
        }
        // Subdivide.
        objects.sort_by(comparator);
        // Use the Surface Area Heuristic (SAH) to determine where to partition
        // the children.
        let mut root_bound = objects[0].bounding_box();
        for sphere in &objects[1..] {
            root_bound = root_bound.merge(&sphere.bounding_box());
        }

        let (best_split, best_cost) = (1..objects.len())
            .map(|split_idx| {
                // Left box
                let mut left = objects[0].bounding_box();
                for sphere in &objects[1..split_idx] {
                    left = left.merge(&sphere.bounding_box());
                }
                // Right box
                let mut right = objects[split_idx].bounding_box();
                for sphere in &objects[split_idx..] {
                    right = right.merge(&sphere.bounding_box());
                }
                let split_cost = TRAVERSAL_COST
                    + left.surface_area() * split_idx as f64 * INTERSECT_COST
                    + right.surface_area() * (objects.len() - split_idx) as f64 * INTERSECT_COST;
                (split_idx, split_cost)
            })
            .min_by_key(|(_, cost)| NonNan::new(*cost).unwrap())
            .unwrap();

        if best_cost > (root_bound.surface_area() * objects.len() as f64 * INTERSECT_COST) {
            // It's cheaper to keep this node as-is instead of splitting.
            return BVHNode::Leaf(BVHLeafNode::new(objects.to_vec()));
        }

        let left = Arc::new(BVHNode::new(&mut objects[..best_split]));
        let right = Arc::new(BVHNode::new(&mut objects[best_split..]));
        let box_left = left.bounding_box();
        let box_right = right.bounding_box();
        let bound = box_left.merge(&box_right);
        BVHNode::Inner(BVHInnerNode {
            left: Some(left),
            right: Some(right),
            bound,
        })
    }
}

impl<H> BVHNode<H>
where
    H: Hittable + Bounded,
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
struct BVHInnerNode<H> {
    left: Option<Arc<BVHNode<H>>>,
    right: Option<Arc<BVHNode<H>>>,
    bound: AABB,
}

impl<H> BVHInnerNode<H>
where
    H: Hittable + Bounded,
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
struct BVHLeafNode<H> {
    objects: Arc<[H]>,
    bound: AABB,
}

impl<H> BVHLeafNode<H>
where
    H: Hittable + Bounded,
{
    fn new(objects: Vec<H>) -> Self {
        let mut bound = objects[0].bounding_box();
        for obj in &objects[1..] {
            bound = bound.merge(&obj.bounding_box());
        }
        Self {
            objects: objects.into(),
            bound,
        }
    }

    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut closest = t_max;
        let mut closest_hit = None;
        for o in &self.objects[..] {
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
