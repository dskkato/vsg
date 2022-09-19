use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

use crate::vertex::Vertex;
use crate::{create_vertices, Instance, InstanceRaw};

use super::{Stim, Visibility};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GratingParams {
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

impl GratingParams {
    pub fn new() -> Self {
        GratingParams {
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
        if self.contrast < 0.5 {
            self.contrast += 0.001;
        } else {
            self.contrast = 0.0;
        }
    }
}

pub struct Grating {
    render_pipeline: wgpu::RenderPipeline,
    // for vertices
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    // for each instance
    instance: Instance,
    instance_buffer: wgpu::Buffer,
    params: GratingParams,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    visible: Visibility,
}

impl Grating {
    pub fn new(
        device: &wgpu::Device,
        proj_bind_group_layout: &wgpu::BindGroupLayout,
        instance: Instance,
    ) -> Self {
        let shader = device.create_shader_module(include_wgsl!("grating.wgsl"));

        let params = GratingParams::new();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_bind_group_layout =
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
                label: Some("params_bind_group_layout"),
            });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
            label: Some("proj_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&proj_bind_group_layout, &params_bind_group_layout],
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

        let (v, idx) = create_vertices(params.diameter);
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

        let instance_data = &[instance].iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let visible = Visibility::Visible;

        Grating {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance,
            instance_buffer,
            params,
            params_buffer,
            params_bind_group,
            visible,
        }
    }

    pub fn update_params(
        &mut self,
        queue: &wgpu::Queue,
        instance: Instance,
        params: GratingParams,
    ) {
        if instance != self.instance {
            self.instance = instance;
            let instance_raw = instance.to_raw();
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&[instance_raw]),
            );
        }

        self.params = GratingParams {
            tick: self.params.tick,
            ..params
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }
}

impl Stim for Grating {
    fn update(&mut self, queue: &wgpu::Queue) {
        self.params.tick();

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
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
                rpass.set_bind_group(1, &self.params_bind_group, &[]);
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
