use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

use crate::vertex::Vertex;
use crate::{Instance, InstanceRaw};

use super::{Stim, Visibility};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RectParams {
    pub a: f32,
    pub b: f32,
}

impl RectParams {
    fn new() -> Self {
        RectParams { a: 0.2, b: 0.3 }
    }
}

fn create_vertices(a: f32, b: f32) -> (Vec<Vertex>, Vec<u16>) {
    let v = vec![
        Vertex {
            // top-left
            position: [-a, b, 0.0],
        },
        Vertex {
            // bottom-left
            position: [-a, -b, 0.0],
        },
        Vertex {
            // bottom-right
            position: [a, -b, 0.0],
        },
        Vertex {
            // top-right
            position: [a, b, 0.0],
        },
    ];
    let idx = vec![0, 1, 2, 0, 2, 3];

    (v, idx)
}

pub struct Rect {
    render_pipeline: wgpu::RenderPipeline,
    // for vertices
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    // for each instance
    instance: Instance,
    instance_buffer: wgpu::Buffer,
    visible: Visibility,
}

impl Rect {
    pub fn new(
        device: &wgpu::Device,
        proj_bind_group_layout: &wgpu::BindGroupLayout,
        instance: Instance,
    ) -> Self {
        let shader = device.create_shader_module(include_wgsl!("rect.wgsl"));

        let rectp_uniform = RectParams::new();

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&proj_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let (v, idx) = create_vertices(rectp_uniform.a, rectp_uniform.b);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = idx.len() as u32;

        let instance_data = [instance.to_raw()];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let visible = Visibility::Visible;

        Rect {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance,
            instance_buffer,
            visible,
        }
    }
}

impl Stim for Rect {
    fn update(&mut self, queue: &wgpu::Queue) {
        let instance_data = [self.instance]
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    fn draw<'a, 'encoder>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'encoder>,
        proj_bind_group: &'encoder wgpu::BindGroup,
    ) where
        'a: 'encoder,
    {
        match self.visible {
            Visibility::Visible => {
                rpass.set_pipeline(&self.render_pipeline);
                rpass.set_bind_group(0, proj_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
            Visibility::Hidden => {}
        }
    }

    fn visibility(&mut self, visible: Visibility) {
        self.visible = visible;
    }
}
