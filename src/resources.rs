use crate::texture;

pub fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new("res").join(file_name);
    let data = std::fs::read(path)?;

    Ok(data)
}

pub fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}
