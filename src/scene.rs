use std::sync::Arc;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;

use crate::surfaces;

use anyhow::{
    anyhow,
    Context,
    Error,
};
use serde::de::{
    self,
    Deserializer,
};
use serde::Deserialize;
use serde_yaml;

use crate::geom::{
    Point3,
    Vec3,
};

pub fn load_scene(path: &str, aspect_ratio: f64) -> Result<(crate::Scene, crate::Camera), Error> {
    let file = File::open(path).with_context(|| format!("could not load scene file {}", path))?;
    let reader = BufReader::new(file);

    let config = serde_yaml::from_reader::<_, Config>(reader)?;

    // load all the meshes
    let mut meshes = HashMap::new();
    for mesh_def in &config.scene.meshes {
        // FIXME: scale should not be tied to the object.
        let scale = mesh_def.scale.unwrap_or(1.0);
        let mesh = load_mesh(mesh_def.path.to_str().unwrap(), scale)
            .with_context(|| format!("could not load mesh {}", mesh_def.name))?;

        if meshes.contains_key(&mesh_def.name) {
            Err(anyhow!("duplicate mesh entry for name '{}'", mesh_def.name))?;
        }
        meshes.insert(mesh_def.name.clone(), mesh);
    }

    let mut builder = crate::SceneBuilder::new();
    for surface in &config.scene.surfaces {
        match surface {
            Surface::Mesh { name, material, .. } => {
                let mesh = meshes
                    .get(name)
                    .ok_or_else(|| anyhow!("mesh with name '{}' not found", name))?;
                mesh.clone()
                    .triangles(material.into())
                    .for_each(|t| builder.add(t));
            }
            Surface::Sphere {
                radius,
                position,
                material,
            } => {
                // FIXME: need to support generic translations
                //
                // A sublety here is that the bounding volumes need to have the
                // actual coordinates.
                let center = (*position).into();
                let sphere = crate::surfaces::Sphere::new(center, *radius, material.into());
                builder.add(sphere);
            }
        }
    }
    Ok((builder.build(), config.camera.build(aspect_ratio)))
}

fn load_mesh(path: &str, scale: f64) -> anyhow::Result<Arc<surfaces::Mesh>> {
    let (models, materials) =
        tobj::load_obj(path, true).context("could not load mesh from file")?;

    if models.len() != 1 {
        return Err(anyhow!("expected exactly one model, got: {}", models.len()));
    }
    if materials.len() > 0 {
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

    // Compute all the vertices along with the model origin.
    //
    // NOTE: This origin is computed by using the center of the bounding box.
    // Alternatively, we could find the center-of-mass by taking the weighted
    // average of each face's center.
    let mut vertices = positions
        .chunks_exact(3)
        .map(|chunk| Point3::new(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64))
        .collect::<Vec<_>>();
    let mut min_v = Point3::new(
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
        ::std::f64::INFINITY,
    );
    let mut max_v = Point3::default();
    for v in &vertices {
        max_v = max_v.max_pointwise(&v);
        min_v = min_v.min_pointwise(&v);
    }
    // let mesh_origin = (min_v + 0.5 * (max_v - min_v)).into();

    // let rotate_about_x =
    //     crate::geom::UnitQuaternion::rotation(Vec3::ihat(), (90.0f64).to_radians());
    // let rotate_about_y =
    //     crate::geom::UnitQuaternion::rotation(Vec3::jhat(), (-100.0f64).to_radians());
    // let rotation = rotate_about_y * rotate_about_x;
    //
    let indices = indices.iter().map(|u| *u as usize).collect::<Vec<_>>();
    for v in vertices.iter_mut() {
    //     let rotated = rotation.rotate_point(*v - mesh_origin) + mesh_origin;
        *v = scale * *v;
    }
    return if normals.len() > 0 {
        // Normals were included, yay
        let normals = normals
            .chunks_exact(3)
            .map(|chunk| Vec3::new(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64))
            .collect();
        Ok(Arc::new(surfaces::Mesh::new_with_normals(indices, vertices, normals)))
    } else {
        // We need to compute the normals ourselves :(
        Ok(Arc::new(surfaces::Mesh::new(indices, vertices)))
    };
}

#[derive(Deserialize, Debug)]
struct Config {
    camera: CameraConfig,
    scene: SceneConfig,
}

#[derive(Deserialize, Debug)]
struct CameraConfig {
    fov: f64,
    from: [f64; 3],
    towards: [f64; 3],
    focus_distance: Option<f64>,
    aperture: Option<f64>,
}

impl CameraConfig {
    fn build(self, aspect_ratio: f64) -> crate::camera::Camera {
        let mut builder = crate::camera::Camera::builder(self.fov, aspect_ratio)
            .from(self.from.into())
            .towards(self.towards.into());
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
    meshes: Vec<MeshDefinition>,
    surfaces: Vec<Surface>,
}

#[derive(Deserialize, Debug)]
struct MeshDefinition {
    name: String,
    scale: Option<f64>,
    path: PathBuf,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Surface {
    Sphere {
        radius: f64,
        position: [f64; 3],
        material: Material,
    },
    Mesh {
        name: String,
        material: Material,
        transform: Vec<Transform>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Transform {
    Scale { factor: f64 },
    Rotate { axis: RotationAxis, angle: f64 },
    Translate { v: [f64; 3] },
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum RotationAxis {
    Basis(Basis),
    General([f64; 3]),
}

impl RotationAxis {
    fn to_vec(&self) -> Vec3 {
        match *self {
            Self::Basis(Basis::X) => Vec3::ihat(),
            Self::Basis(Basis::Y) => Vec3::jhat(),
            Self::Basis(Basis::Z) => Vec3::khat(),
            Self::General(v) => v.into(),
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
#[serde(tag = "type")]
enum Material {
    #[serde(rename = "lambertian")]
    Lambertian { albedo: Albedo },
    #[serde(rename = "dielectric")]
    Dielectric { index: f64 },
    #[serde(rename = "metal")]
    Metal { albedo: Albedo, fuzz: f64 },
}

impl Into<crate::material::Material> for &Material {
    fn into(self) -> crate::material::Material {
        match *self {
            Material::Lambertian { ref albedo } => {
                crate::material::Material::lambertian(albedo.0.into())
            }
            Material::Dielectric { index } => crate::material::Material::dielectric(index),
            Material::Metal { ref albedo, fuzz } => {
                crate::material::Material::metal(albedo.0.into(), fuzz)
            }
        }
    }
}

#[derive(Debug)]
struct Albedo([f64; 3]);

impl FromStr for Albedo {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Should look like #ffffff
        if s.len() != 6 {
            Err(anyhow!("expected color format #aabbcc, got {}", s))?;
        }
        let parsed = u32::from_str_radix(&s, 16).context("could not parse color")?;
        let bytes: [u8; 4] = parsed.to_be_bytes();

        Ok(Albedo([
            bytes[1] as f64 / 256.0,
            bytes[2] as f64 / 256.0,
            bytes[3] as f64 / 256.0,
        ]))
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
