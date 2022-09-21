pub mod grating;
pub mod picture;
pub mod rdk;
pub mod rect;

pub enum Visibility {
    Visible,
    Hidden,
}

pub trait Stim {
    fn update(&mut self, _queue: &wgpu::Queue) {}
    fn reset(&mut self, _queue: &wgpu::Queue) {}
    fn start(&mut self, _queue: &wgpu::Queue) {}

    fn draw<'a, 'encoder>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'encoder>,
        proj_bind_group: &'encoder wgpu::BindGroup,
    ) where
        'a: 'encoder;

    fn visibility(&mut self, visible: Visibility);
}
