pub struct Projection {
    pub aspect: f32,
}

impl Projection {
    pub fn new(aspect: f32) -> Projection {
        Projection { aspect }
    }

    pub fn to_raw(&self) -> ProjectionUniform {
        // cgmath::ortho(-1.0, 1.0, -self.aspect, self.aspect, 1.0, -1.0); // これに相当するMatrix4
        ProjectionUniform {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0 / self.aspect, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjectionUniform {
    view: [[f32; 4]; 4],
}
