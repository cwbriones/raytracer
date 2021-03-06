/// Utilities and bindings for loading a scene from YAML configuration.
///
/// This module contains serde bindings and wrappers for defining the scene,
/// opting for explicit conversion here rather than sprinkle #[derive(Deserialize)]
/// throughout the code. The sole exception to this is for some types in geom,
/// since those are unlikely to change.
use std::fmt;
use std::fs::File;
use std::io::BufReader;
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
pub fn load_scene(path: &str, aspect_ratio: f64) -> anyhow::Result<(Scene, Camera)> {
    let file = File::open(path).with_context(|| format!("could not load scene file {}", path))?;
    let reader = BufReader::new(file);

    let config = serde_yaml::from_reader::<_, Config>(reader)?;

    if config.scene.surfaces.is_empty() {
        return Err(anyhow!("scene '{}' is empty", path));
    }

    let mut builder = Scene::builder();
    for surface in &config.scene.surfaces {
        match surface {
            Surface::Mesh {
                path,
                material,
                transform,
            } => {
                let mesh = load_mesh(path, transform).with_context(|| {
                    format!("could not load mesh at {}", path.to_string_lossy())
                })?;
                mesh.triangles(material.into()).for_each(|t| builder.add(t));
            }
            Surface::Sphere {
                radius,
                position,
                material,
            } => {
                // FIXME: need to support generic translations
                let sphere = crate::surfaces::Sphere::new(*position, *radius, material.into());
                builder.add(sphere);
            }
        }
    }
    Ok((builder.build(), config.camera.build(aspect_ratio)))
}

fn load_mesh(
    path: &std::path::Path,
    transform: &[Transform],
) -> anyhow::Result<Arc<surfaces::Mesh>> {
    let (models, materials) =
        tobj::load_obj(path, true).context("could not load mesh from file")?;

    if models.len() != 1 {
        return Err(anyhow!("expected exactly one model, got: {}", models.len()));
    }
    if !materials.is_empty() {
        eprintln!(
            "warning: {} materials found in OBJ file. Materials are not supported.",
            materials.len()
        );
    }
    let model = models.into_iter().next().unwrap(); // we already checked len above.

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

    // Recenter the model
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

/// Compute the center of the mesh.
///
/// NOTE: This origin is computed by using the center of the bounding box.
/// Alternatively, we could find the center-of-mass by taking the weighted
/// average of each face's center.
fn compute_mesh_center(vertices: &[Point3]) -> Vec3 {
    let mut min_v = Point3::new(
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
    );
    let mut max_v = Point3::default();
    for v in vertices {
        max_v = max_v.max_pointwise(&v);
        min_v = min_v.min_pointwise(&v);
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
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Surface {
    Sphere {
        radius: f64,
        position: Point3,
        material: Material,
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
}

impl From<&Material> for crate::material::Material {
    fn from(material: &Material) -> Self {
        match *material {
            Material::Lambertian { ref albedo } => crate::material::Material::lambertian(albedo.0),
            Material::Dielectric { index } => crate::material::Material::dielectric(index),
            Material::Metal { ref albedo, fuzz } => {
                crate::material::Material::metal(albedo.0, fuzz)
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
        let parsed = u32::from_str_radix(&s, 16).context("could not parse hex color")?;
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
