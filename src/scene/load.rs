/// Utilities and bindings for loading a scene from YAML configuration.
///
/// This module contains serde bindings and wrappers for defining the scene,
/// opting for explicit conversion here rather than sprinkle #[derive(Deserialize)]
/// throughout the code. The sole exception to this is for some types in geom,
/// since those are unlikely to change.
use std::fmt;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{
    anyhow,
    Context,
};
use serde::de::{
    self,
    Deserializer,
};
use serde::Deserialize;

use crate::camera::Camera;
use crate::geom::UnitQuaternion;
use crate::geom::{
    Point3,
    Vec3,
};
use crate::scene::Scene;
use crate::surfaces;

/// Load a scene from the given path.
///
/// The camera will be configured with the given aspect ratio.
pub fn load_scene<P: AsRef<Path>>(
    path: P,
    aspect_ratio: f64,
    use_bvh: bool,
) -> anyhow::Result<(Scene, Camera)> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let config = serde_yaml::from_reader::<_, Config>(reader)?;

    if config.scene.surfaces.is_empty() {
        return Err(anyhow!("scene is empty"));
    }

    let mut builder = Scene::builder();
    builder.set_background(config.scene.background);
    builder.set_use_bvh(use_bvh);
    for surface in &config.scene.surfaces {
        match surface {
            Surface::Mesh {
                path: relpath,
                material,
                transform,
            } => {
                let mesh_path = path
                    .parent()
                    .map(|p| p.join(relpath))
                    .ok_or_else(|| anyhow!("invalid mesh path: '{}'", relpath.display()))?;
                let mesh = load_mesh(&mesh_path, transform)
                    .with_context(|| format!("could not load mesh at '{}'", path.display()))?;
                mesh.triangles(material.into()).for_each(|t| builder.add(t));
            }
            Surface::Sphere {
                radius,
                position,
                material,
            } => {
                // FIXME: need to support generic translations
                let sphere = surfaces::Sphere::new(*position, *radius, material.into());
                builder.add(sphere);
            }
            Surface::Quad {
                position,
                sides,
                material,
            } => {
                let (u, v) = sides;
                let quad = surfaces::Quad::new(*position, *u, *v, material.into());
                builder.add(quad);
            }
            Surface::Box {
                corners,
                material,
                transform,
            } => {
                let (a, b) = corners;
                let min = a.min_pointwise(b);
                let max = a.max_pointwise(b);

                let dx = Vec3::new(max.x() - min.x(), 0.0, 0.0);
                let dy = Vec3::new(0.0, max.y() - min.y(), 0.0);
                let dz = Vec3::new(0.0, 0.0, max.z() - min.z());

                let material: crate::material::Material = material.into();
                let quads = [
                    // front
                    crate::surfaces::Quad::new(
                        Point3::new(min.x(), min.y(), max.z()),
                        dx,
                        dy,
                        material.clone(),
                    ),
                    // right
                    crate::surfaces::Quad::new(
                        Point3::new(max.x(), min.y(), max.z()),
                        dz.negate(),
                        dy,
                        material.clone(),
                    ),
                    // back
                    crate::surfaces::Quad::new(
                        Point3::new(max.x(), min.y(), min.z()),
                        dx.negate(),
                        dy,
                        material.clone(),
                    ),
                    // left
                    crate::surfaces::Quad::new(
                        Point3::new(min.x(), min.y(), min.z()),
                        dz,
                        dy,
                        material.clone(),
                    ),
                    // top
                    crate::surfaces::Quad::new(
                        Point3::new(min.x(), max.y(), max.z()),
                        dx,
                        dz.negate(),
                        material.clone(),
                    ),
                    // bottom
                    crate::surfaces::Quad::new(
                        Point3::new(min.x(), min.y(), min.z()),
                        dx,
                        dz,
                        material,
                    ),
                ];
                let points = quads
                    .iter()
                    .flat_map(|quad| [quad.q, quad.q + quad.u + quad.v])
                    .collect::<Vec<_>>();
                let origin = compute_mesh_center(&points);
                for q in &quads {
                    builder.add(transform_quad(q, transform, origin));
                }
            }
        }
    }
    Ok((builder.build(), config.camera.build(aspect_ratio)))
}

fn transform_quad(
    quad: &surfaces::Quad,
    transforms: &[Transform],
    mut origin: Vec3,
) -> surfaces::Quad {
    let mut q1 = quad.q;
    let mut q2 = quad.q + quad.u;
    let mut q3 = quad.q + quad.v;
    for t in transforms {
        match *t {
            Transform::Scale { factor } => {
                for v in [&mut q1, &mut q2, &mut q3] {
                    *v = factor * (*v - origin) + origin;
                }
            }
            Transform::Rotate { ref axis, angle } => {
                let axis = axis.to_vec();
                let rotation = UnitQuaternion::rotation(axis, angle.to_radians());
                for v in [&mut q1, &mut q2, &mut q3] {
                    *v = rotation.rotate_point(*v - origin) + origin;
                }
            }
            Transform::Translate { dir } => {
                for v in [&mut q1, &mut q2, &mut q3] {
                    *v += dir
                }
                origin += dir;
            }
            Transform::TranslateTo { dest } => {
                let dir = dest - origin;
                for v in [&mut q1, &mut q2, &mut q3] {
                    *v += dir
                }
                origin += dir;
            }
        }
    }
    surfaces::Quad::new(q1, q2 - q1, q3 - q1, quad.material.clone())
}

fn load_mesh<P: AsRef<Path>>(
    path: P,
    transform: &[Transform],
) -> anyhow::Result<Arc<surfaces::Mesh>> {
    let mut reader = reader_at(path)?;
    let (models, materials) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            ..Default::default()
        },
        // Since we don't support materials right now, just fail immediately.
        |_| Err(tobj::LoadError::GenericFailure),
    )
    .context("could not load mesh from file")?;

    if let Ok(m) = materials {
        if !m.is_empty() {
            eprintln!(
                "warning: {} materials found in OBJ file. Materials are not supported.",
                m.len()
            );
        }
    }
    let nmodels = models.len();
    let model = models
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("expected exactly one model, got: {}", nmodels))?; // we already checked len above.

    eprintln!("loaded mesh: {}", model.name);
    let ::tobj::Mesh {
        positions,
        normals,
        indices,
        ..
    } = model.mesh;
    eprintln!("vertices: {}", positions.len());
    eprintln!(
        "indices: {} ({} triangles)",
        indices.len(),
        indices.len() / 3
    );
    eprintln!("normals: {}", normals.len());
    if normals.len() % 3 != 0 {
        return Err(anyhow!(
            "number of normals not divisible by 3: {}",
            normals.len()
        ));
    }
    if indices.len() % 3 != 0 {
        return Err(anyhow!(
            "number of indices not divisible by 3: {}",
            indices.len()
        ));
    }

    let mut vertices = positions
        .chunks_exact(3)
        .map(|chunk| Point3::new(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64))
        .collect::<Vec<_>>();
    let indices = indices.iter().map(|u| *u as usize).collect::<Vec<_>>();

    // Recompute the vertices relative to the model origin
    let mut mesh_origin = compute_mesh_center(&vertices);
    for v in vertices.iter_mut() {
        *v = *v - mesh_origin;
    }
    for t in transform {
        match *t {
            Transform::Scale { factor } => {
                for v in vertices.iter_mut() {
                    *v = factor * *v;
                }
                mesh_origin = compute_mesh_center(&vertices);
            }
            Transform::Rotate { ref axis, angle } => {
                let axis = axis.to_vec();
                let rotation = UnitQuaternion::rotation(axis, angle.to_radians());
                for v in vertices.iter_mut() {
                    *v = rotation.rotate_point(*v - mesh_origin) + mesh_origin;
                }
            }
            Transform::Translate { dir } => {
                for v in vertices.iter_mut() {
                    *v += dir;
                }
                mesh_origin += dir;
            }
            Transform::TranslateTo { dest } => {
                let dir = dest - mesh_origin;
                for v in vertices.iter_mut() {
                    *v += dir;
                }
                mesh_origin += dir;
            }
        }
    }
    return if normals.is_empty() {
        // We need to compute the normals ourselves :(
        Ok(Arc::new(surfaces::Mesh::new(indices, vertices)))
    } else {
        // Normals were included, yay
        let normals = normals
            .chunks_exact(3)
            .map(|chunk| Vec3::new(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64))
            .collect();
        Ok(Arc::new(surfaces::Mesh::new_with_normals(
            indices, vertices, normals,
        )))
    };
}

fn reader_at<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>, io::Error> {
    let path = path.as_ref();
    let file = File::open(path)?;
    if let Some("gz") = path.extension().and_then(|s| s.to_str()) {
        // Unfortunately, the GzDecoder doesn't implement BufRead itself
        // so we need two levels of buffering (input buffer, output buffer).
        let reader = flate2::read::GzDecoder::new(file);
        return Ok(Box::new(BufReader::new(reader)));
    }
    Ok(Box::new(BufReader::new(file)))
}

/// Compute the center of the mesh.
///
/// NOTE: This origin is computed by using the center of the bounding box.
/// Alternatively, we could find the center-of-mass by taking the weighted
/// average of each face's center.
fn compute_mesh_center<'a, I: IntoIterator<Item = &'a Point3>>(vertices: I) -> Vec3 {
    let mut min_v = Point3::new(
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
    );
    let mut max_v = Point3::default();
    for v in vertices {
        max_v = max_v.max_pointwise(v);
        min_v = min_v.min_pointwise(v);
    }
    (min_v + 0.5 * (max_v - min_v)).into()
}

#[derive(Deserialize, Debug)]
struct Config {
    camera: CameraConfig,
    scene: SceneConfig,
}

#[derive(Deserialize, Debug)]
struct CameraConfig {
    fov: f64,
    from: Point3,
    towards: Point3,
    focus_distance: Option<f64>,
    aperture: Option<f64>,
}

impl CameraConfig {
    fn build(self, aspect_ratio: f64) -> crate::camera::Camera {
        let mut builder = crate::camera::Camera::builder(self.fov, aspect_ratio)
            .from(self.from)
            .towards(self.towards);
        if let Some(ref aperture) = self.aperture {
            builder = builder.aperture(*aperture);
        }
        if let Some(ref focus_distance) = self.focus_distance {
            builder = builder.focus_dist(*focus_distance);
        }
        builder.build()
    }
}

#[derive(Deserialize, Debug)]
struct SceneConfig {
    surfaces: Vec<Surface>,
    background: Vec3,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Surface {
    Sphere {
        radius: f64,
        position: Point3,
        material: Material,
    },
    Quad {
        position: Point3,
        sides: (Vec3, Vec3),
        material: Material,
    },
    Box {
        corners: (Point3, Point3),
        material: Material,
        #[serde(default)]
        transform: Vec<Transform>,
    },
    Mesh {
        path: PathBuf,
        material: Material,
        #[serde(default)]
        transform: Vec<Transform>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Transform {
    Scale { factor: f64 },
    Rotate { axis: RotationAxis, angle: f64 },
    Translate { dir: Vec3 },
    TranslateTo { dest: Vec3 },
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum RotationAxis {
    Basis(Basis),
    General(Vec3),
}

impl RotationAxis {
    fn to_vec(&self) -> Vec3 {
        match *self {
            Self::Basis(Basis::X) => Vec3::ihat(),
            Self::Basis(Basis::Y) => Vec3::jhat(),
            Self::Basis(Basis::Z) => Vec3::khat(),
            Self::General(v) => v,
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
enum Basis {
    X,
    Y,
    Z,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Material {
    Lambertian { albedo: Albedo },
    Dielectric { index: f64 },
    Metal { albedo: Albedo, fuzz: f64 },
    DiffuseLight { albedo: Albedo },
}

impl From<&Material> for crate::material::Material {
    fn from(material: &Material) -> Self {
        match *material {
            Material::Lambertian { ref albedo } => crate::material::Material::lambertian(albedo.0),
            Material::Dielectric { index } => crate::material::Material::dielectric(index),
            Material::Metal { ref albedo, fuzz } => {
                crate::material::Material::metal(albedo.0, fuzz)
            }
            Material::DiffuseLight { ref albedo } => {
                crate::material::Material::diffuse_light(albedo.0)
            }
        }
    }
}

#[derive(Debug)]
struct Albedo(Vec3);

impl FromStr for Albedo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 6 {
            return Err(anyhow!("expected hex color format aabbcc, got {}", s));
        }
        let parsed = u32::from_str_radix(s, 16).context("could not parse hex color")?;
        let bytes: [u8; 4] = parsed.to_be_bytes();

        Ok(Albedo(Vec3::new(
            bytes[1] as f64 / 256.0,
            bytes[2] as f64 / 256.0,
            bytes[3] as f64 / 256.0,
        )))
    }
}

impl<'de> Deserialize<'de> for Albedo {
    fn deserialize<D>(deserializer: D) -> Result<Albedo, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AlbedoVisitor(std::marker::PhantomData<Albedo>);

        impl<'de> de::Visitor<'de> for AlbedoVisitor {
            type Value = Albedo;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("string or array of floats")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Albedo::from_str(value).map_err(serde::de::Error::custom)
            }

            fn visit_seq<S>(self, visitor: S) -> Result<Self::Value, S::Error>
            where
                S: de::SeqAccess<'de>,
            {
                let inner =
                    Deserialize::deserialize(de::value::SeqAccessDeserializer::new(visitor))?;
                Ok(Albedo(inner))
            }
        }

        deserializer.deserialize_any(AlbedoVisitor(std::marker::PhantomData))
    }
}
