use std::sync::Arc;

use crate::surfaces::Sphere;
use crate::trace::{
    Hit,
    Ray,
    AABB,
};
use crate::util::NonNan;

/// The cost of computing an intersection given a ray.
const INTERSECT_COST: f64 = 1.0;

/// The cost of traversing from a parent to child node in the tree.
const TRAVERSAL_COST: f64 = 2.0;

#[derive(Clone)]
pub struct BVH {
    root: BVHNode,
}

impl BVH {
    pub fn new(spheres: &mut [Sphere]) -> Self {
        BVH {
            root: BVHNode::new(spheres),
        }
    }

    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        self.root.hit(ray, t_min, t_max)
    }
}

#[derive(Clone)]
enum BVHNode {
    Inner(BVHInnerNode),
    Leaf(BVHLeafNode),
}

impl BVHNode {
    fn new(spheres: &mut [Sphere]) -> Self {
        // Choose the axis
        let axis = (0usize..3)
            .max_by_key(|i| {
                // Choose the axis that has the widest span of centroids.
                //
                // This is the distance between the rightmost and leftmost
                // bounding boxes on a given axis.
                let mut min = f64::INFINITY;
                let mut max = 0.0;
                for sphere in spheres.iter() {
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
        let comparator = |a: &Sphere, b: &Sphere| {
            let bba = a.bounding_box();
            let bbb = b.bounding_box();
            bba.min()
                .get(axis)
                .partial_cmp(&bbb.min().get(axis))
                .unwrap()
        };
        let min_split_len = 4;
        if spheres.len() <= min_split_len {
            return BVHNode::Leaf(BVHLeafNode::new(spheres.to_vec()));
        }
        // Subdivide.
        spheres.sort_by(comparator);
        // Use the Surface Area Heuristic (SAH) to determine where to partition
        // the children.
        let mut root_bound = spheres[0].bounding_box();
        for sphere in &spheres[1..] {
            root_bound = root_bound.merge(&sphere.bounding_box());
        }

        let (best_split, best_cost) = (1..spheres.len())
            .map(|split_idx| {
                // Left box
                let mut left = spheres[0].bounding_box();
                for sphere in &spheres[1..split_idx] {
                    left = left.merge(&sphere.bounding_box());
                }
                // Right box
                let mut right = spheres[split_idx].bounding_box();
                for sphere in &spheres[split_idx..] {
                    right = right.merge(&sphere.bounding_box());
                }
                let split_cost = TRAVERSAL_COST
                    + left.surface_area() * split_idx as f64 * INTERSECT_COST
                    + right.surface_area() * (spheres.len() - split_idx) as f64 * INTERSECT_COST;
                (split_idx, split_cost)
            })
            .min_by_key(|(_, cost)| NonNan::new(*cost).unwrap())
            .unwrap();

        if best_cost > (root_bound.surface_area() * spheres.len() as f64 * INTERSECT_COST) {
            // It's cheaper to keep this node as-is instead of splitting.
            return BVHNode::Leaf(BVHLeafNode::new(spheres.to_vec()));
        }

        let left = Arc::new(BVHNode::new(&mut spheres[..best_split]));
        let right = Arc::new(BVHNode::new(&mut spheres[best_split..]));
        let box_left = left.bounding_box();
        let box_right = right.bounding_box();
        let bound = box_left.merge(&box_right);
        BVHNode::Inner(BVHInnerNode {
            left: Some(left),
            right: Some(right),
            bound,
        })
    }

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
struct BVHInnerNode {
    left: Option<Arc<BVHNode>>,
    right: Option<Arc<BVHNode>>,
    bound: AABB,
}

impl BVHInnerNode {
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
struct BVHLeafNode {
    objects: Arc<[Sphere]>,
    bound: AABB,
}

impl BVHLeafNode {
    fn new(objects: Vec<Sphere>) -> Self {
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