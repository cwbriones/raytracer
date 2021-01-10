# raytracer
[![ci_badge]][actions] [![license_badge]](LICENSE)

A playground for experimenting with raytracing in rust.

This is primarily based on Peter Shirley's [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) book series, with some modifications as I learn about raytracers more generally.

So far I've implemented everything in the first book, with the addition of the BVH acceleration structure as described in [PBR](http://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies.html).

## License

This code is available under the MIT license.

See [LICENSE](LICENSE) for details.

[ci_badge]: https://github.com/cwbriones/raytracer/workflows/ci/badge.svg?branch=master
[actions]: https://github.com/cwbriones/raytracer/actions?query=workflow%3Aci+branch%3Amaster
[license_badge]: https://img.shields.io/github/license/cwbriones/raytracer?color=blue
