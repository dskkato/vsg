use cgmath::prelude::*;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use std::{iter, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod resources;
mod texture;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GratingParamsUniform {
    sf: f32,
    tf: f32,
    phase: f32,
    contrast: f32,
    tick: f32,
    diameter: f32,
    sigma: f32, // Gaussian window envelop, if sigma < 0.0, apply no window
    has_texture: u32,
    color: [f32; 4],
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
            has_texture: 0,
            color: [0.5, 0.5, 0.5, 1.0],
        }
    }

    fn tick(&mut self) {
        self.tick += 1.0 / 60.0;
        self.phase = self.tf * self.tick;
    }
}

struct Projection {
    pub aspect: f32,
}

impl Projection {
    fn new(aspect: f32) -> Projection {
        Projection { aspect }
    }

    fn to_raw(&self) -> ProjectionUniform {
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
struct ProjectionUniform {
    view: [[f32; 4]; 4],
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We don't have to do this in code though.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

struct App {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    // view
    proj: Projection,
    proj_buffer: wgpu::Buffer,
    proj_bind_group: wgpu::BindGroup,
    // texture
    #[allow(dead_code)]
    texture: texture::Texture,
    texture_bind_group: wgpu::BindGroup,
    // for vertices
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    // grating params
    gratingp_uniform: GratingParamsUniform,
    gratingp_buffer: wgpu::Buffer,
    gratingp_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    // imgui
    pub imgui: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    last_frame: Instant,
    last_cursor: Option<MouseCursor>,
}

fn create_vertices(d: f32) -> (Vec<Vertex>, Vec<u16>) {
    let v = vec![
        Vertex {
            // top-left
            position: [-d, d, 0.0],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            // bottom-left
            position: [-d, -d, 0.0],
            tex_coords: [0.0, 1.0],
        },
        Vertex {
            // bottom-right
            position: [d, -d, 0.0],
            tex_coords: [1.0, 1.0],
        },
        Vertex {
            // top-right
            position: [d, d, 0.0],
            tex_coords: [1.0, 0.0],
        },
    ];
    let idx = vec![0, 1, 2, 0, 2, 3];

    (v, idx)
}

impl App {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::DX12);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        // Set up dear imgui
        let hidpi_factor = window.scale_factor();
        let mut imgui = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );
        imgui.set_ini_filename(None);

        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

        //
        // Set up dear imgui wgpu renderer
        //
        let renderer_config = RendererConfig {
            texture_format: config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut imgui, &device, &queue, renderer_config);

        let last_frame = std::time::Instant::now();

        let last_cursor = None;

        let fname = "shaders/shader.wgsl";
        let wgsl = std::fs::read_to_string(fname).unwrap();

        let texture =
            resources::load_texture("macaque.jpg", &device, &queue).expect("Failed to load image.");

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

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

        let proj = Projection::new(size.width as f32 / size.height as f32);
        let proj_uniform = proj.to_raw();
        let proj_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Proj Buffer"),
            contents: bytemuck::cast_slice(&[proj_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let proj_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("proj_bind_group_layout"),
            });

        let proj_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &proj_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: proj_buffer.as_entire_binding(),
            }],
            label: Some("proj_bind_group"),
        });

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &gratingp_bind_group_layout,
                    &proj_bind_group_layout,
                    &texture_bind_group_layout,
                ],
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
                    format: config.format,
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

        App {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            proj,
            proj_buffer,
            proj_bind_group,
            vertex_buffer,
            index_buffer,
            num_indices,
            texture,
            texture_bind_group,
            gratingp_uniform,
            gratingp_buffer,
            gratingp_bind_group,
            instances,
            instance_buffer,
            imgui,
            platform,
            renderer,
            last_frame,
            last_cursor,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.proj = Projection::new(new_size.height as f32 / new_size.width as f32);
            let proj_uniform = self.proj.to_raw();
            self.queue
                .write_buffer(&self.proj_buffer, 0, bytemuck::cast_slice(&[proj_uniform]));
        }
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        // match event {
        //     WindowEvent::CursorMoved { position, .. } => {
        //         self.instances[0].position.x =
        //             ((position.x / self.size.width as f64) * 2.0 - 1.0) as f32;
        //         self.instances[0].position.y = ((position.y / self.size.height as f64) * (-2.0)
        //             + 1.0) as f32
        //             * self.proj.aspect;
        //         false
        //     }
        //     _ => false,
        // }
        false
    }

    fn update(&mut self) {
        self.gratingp_uniform.tick();

        self.queue.write_buffer(
            &self.gratingp_buffer,
            0,
            bytemuck::cast_slice(&[self.gratingp_uniform]),
        );

        let instance_data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // just after flip swap chain
        let delta_s = self.last_frame.elapsed();
        let now = Instant::now();
        self.imgui.io_mut().update_delta_time(now - self.last_frame);
        self.last_frame = now;

        // imgui part
        self.platform
            .prepare_frame(self.imgui.io_mut(), window)
            .expect("Failed to prepare frame");
        let ui = self.imgui.frame();

        // make copy so that color update occurs at the same time between background and shader's one.
        let color = self.gratingp_uniform.color;

        {
            let window = imgui::Window::new("VSG");
            window
                .size([300.0, 120.0], Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text("VSG");
                    ui.separator();
                    ColorEdit::new("Color", &mut self.gratingp_uniform.color).build(&ui);

                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                    ui.text(format!("Frametime: {:.1}", delta_s.as_secs_f64() * 1000.0));
                });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if self.last_cursor != ui.mouse_cursor() {
            self.last_cursor = ui.mouse_cursor();
            self.platform.prepare_render(&ui, window);
        }

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: color[0] as f64,
                            g: color[1] as f64,
                            b: color[2] as f64,
                            a: color[3] as f64,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.gratingp_bind_group, &[]);
            render_pass.set_bind_group(1, &self.proj_bind_group, &[]);
            render_pass.set_bind_group(2, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as u32);

            self.renderer
                .render(ui.render(), &self.queue, &self.device, &mut render_pass)
                .expect("Rendering failed");
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("VSG")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    let mut app = App::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !app.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            app.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &mut so w have to dereference it twice
                            app.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
                window.request_redraw();
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                app.update();
                match app.render(&window) {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        app.resize(app.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }

        app.platform
            .handle_event(app.imgui.io_mut(), &window, &event);
    });
}
