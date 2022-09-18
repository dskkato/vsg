use cgmath::prelude::*;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

use crate::vertex::Vertex;
use crate::{create_vertices, Instance, InstanceRaw};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GratingParamsUniform {
    pub sf: f32,
    pub tf: f32,
    pub phase: f32,
    pub contrast: f32,
    pub tick: f32,
    pub diameter: f32,
    pub sigma: f32, // Gaussian window envelop, if sigma < 0.0, apply no window
    pub _padding: f32,
    pub color: [f32; 4],
}

impl GratingParamsUniform {
    fn new() -> Self {
        GratingParamsUniform {
            sf: 5.0,
            tf: 1.0,
            phase: 0.0,
            contrast: 0.3,
            tick: 0.0,
            diameter: 0.6,
            sigma: 0.15,
            _padding: 0.0,
            color: [0.5, 0.5, 0.5, 1.0],
        }
    }

    pub fn tick(&mut self) {
        self.tick += 1.0 / 60.0;
        self.phase = self.tf * self.tick;
    }
}

pub struct Grating {
    render_pipeline: wgpu::RenderPipeline,
    gratingp_uniform: GratingParamsUniform,
    gratingp_buffer: wgpu::Buffer,
    gratingp_bind_group: wgpu::BindGroup,
    // for vertices
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    // for each instance
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

impl Grating {
    pub fn new(
        device: &wgpu::Device,
        proj_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(include_wgsl!("grating.wgsl"));

        let gratingp_uniform = GratingParamsUniform::new();
        let gratingp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grating Params Buffer"),
            contents: bytemuck::cast_slice(&[gratingp_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gratingp_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("gratingp_bind_group_layout"),
            });

        let gratingp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gratingp_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gratingp_buffer.as_entire_binding(),
            }],
            label: Some("gratingp_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&gratingp_bind_group_layout, &proj_bind_group_layout],
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

        let (v, idx) = create_vertices(gratingp_uniform.diameter);
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

        let instances = (-1..2)
            .step_by(2)
            .map(|x| {
                let position = cgmath::Vector3 {
                    x: x as f32 / 2.0,
                    y: 0.0,
                    z: 1.0f32,
                };
                let rotation = cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_z(),
                    cgmath::Deg(0.0 * x as f32),
                );
                Instance { position, rotation }
            })
            .collect::<Vec<_>>();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Grating {
            render_pipeline,
            gratingp_uniform,
            gratingp_buffer,
            gratingp_bind_group,
            vertex_buffer,
            index_buffer,
            num_indices,
            instances,
            instance_buffer,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        self.gratingp_uniform.tick();

        queue.write_buffer(
            &self.gratingp_buffer,
            0,
            bytemuck::cast_slice(&[self.gratingp_uniform]),
        );

        let instance_data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    pub fn draw<'a, 'encoder>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'encoder>,
        proj_bind_group: &'encoder wgpu::BindGroup,
    ) where
        'a: 'encoder,
    {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.gratingp_bind_group, &[]);
        rpass.set_bind_group(1, proj_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rpass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as u32);
    }
}