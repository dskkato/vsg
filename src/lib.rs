use std::{iter, time::Instant};

use cgmath::prelude::*;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod projection;
use projection::Projection;

mod resources;
mod texture;

mod stim;
use stim::{
    grating::{Grating, GratingParams},
    rdk::RandomDotKinematogram,
    rect::Rect,
    Stim,
};

mod vertex;
use vertex::Vertex;

#[derive(Copy, Clone, PartialEq)]
pub struct Instance {
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
    bg_color: [f32; 4],
    // view
    proj: Projection,
    proj_buffer: wgpu::Buffer,
    proj_bind_group: wgpu::BindGroup,
    // imgui
    pub imgui: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    last_frame: Instant,
    last_cursor: Option<MouseCursor>,
    // grating
    grating: Grating,
    // rect
    rect: Rect,
    rdk: RandomDotKinematogram,
}

fn create_vertices(d: f32) -> (Vec<Vertex>, Vec<u16>) {
    let v = vec![
        Vertex {
            // top-left
            position: [-d, d, 0.0],
        },
        Vertex {
            // bottom-left
            position: [-d, -d, 0.0],
        },
        Vertex {
            // bottom-right
            position: [d, -d, 0.0],
        },
        Vertex {
            // top-right
            position: [d, d, 0.0],
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

        let bg_color = [0.5, 0.5, 0.5, 1.0];
        let instance = Instance {
            position: cgmath::Vector3 {
                x: -1.0 / 2.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: cgmath::Quaternion::from_axis_angle(
                cgmath::Vector3::unit_z(),
                cgmath::Deg(0.0),
            ),
        };
        let grating = Grating::new(&device, &proj_bind_group_layout, instance);

        let instance = Instance {
            position: cgmath::Vector3 {
                x: 1.0 / 2.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: cgmath::Quaternion::from_axis_angle(
                cgmath::Vector3::unit_z(),
                cgmath::Deg(30.0),
            ),
        };
        let rect = Rect::new(&device, &proj_bind_group_layout, instance);
        let rdk = RandomDotKinematogram::new(&device, &proj_bind_group_layout);

        App {
            surface,
            device,
            queue,
            config,
            size,
            bg_color,
            proj,
            proj_buffer,
            proj_bind_group,
            imgui,
            platform,
            renderer,
            last_frame,
            last_cursor,
            grating,
            rect,
            rdk,
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

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let x = ((position.x / self.size.width as f64) * 2.0 - 1.0) as f32;
                let y = ((position.y / self.size.height as f64) * (-2.0) + 1.0) as f32
                    * self.proj.aspect;
                let instance = Instance {
                    position: cgmath::Vector3 { x, y, z: 0.0 },
                    rotation: cgmath::Quaternion::from_axis_angle(
                        cgmath::Vector3::unit_z(),
                        cgmath::Deg(0.0),
                    ),
                };
                let params = GratingParams::new();
                self.grating.update_params(&self.queue, instance, params);
                false
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::R),
                        ..
                    },
                ..
            } => {
                self.rdk.reset(&self.queue);
                false
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.grating.update(&self.queue);
        self.rdk.update(&self.queue);
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // just after flip swap chain
        let delta_s = self.last_frame.elapsed();
        if delta_s.as_millis() > 20 {
            log::info!("Frame fault.");
        }
        let now = Instant::now();
        self.imgui.io_mut().update_delta_time(now - self.last_frame);
        self.last_frame = now;

        // imgui part
        self.platform
            .prepare_frame(self.imgui.io_mut(), window)
            .expect("Failed to prepare frame");
        let ui = self.imgui.frame();

        // make copy so that color update occurs at the same time between background and shader's one.
        let color = self.bg_color;

        {
            let window = imgui::Window::new("VSG");
            window
                .size([300.0, 120.0], Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text("VSG");
                    ui.separator();
                    ColorEdit::new("Color", &mut self.bg_color).build(&ui);

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

            self.grating.draw(&mut render_pass, &self.proj_bind_group);
            self.rect.draw(&mut render_pass, &self.proj_bind_group);
            self.rdk.draw(&mut render_pass, &self.proj_bind_group);

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
