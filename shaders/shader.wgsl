// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) position: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
}

struct ViewUniform {
    proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> view: ViewUniform;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.clip_position = view.proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.position = model.position.xy;
    out.tex_coords = model.tex_coords;
    return out;
}

struct GratingParams {
    sf: f32, // spatial frequency
    tf: f32, // temporal frequency
    phase: f32,
    contrast: f32,
    tick: f32,
    diameter: f32,
    sigma: f32,
    has_texture: u32,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> t: GratingParams;

@group(2) @binding(0)
var t_texture: texture_2d<f32>;
@group(2) @binding(1)
var s_texture: sampler;

let pi = 3.14159;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (t.has_texture > u32(0)) {
        return textureSample(t_texture, s_texture, in.tex_coords.xy);
    }
    let ctr = vec2<f32>(0.0, 0.0);

    if (distance(in.position, ctr) < t.diameter) {
        let pos = in.position.xy;
        var amp: f32;
        if (t.sigma <= 0.0) {
            amp = 1.0;
        } else {
            amp = exp(-(pow(pos.x, 2.0) + pow(pos.y, 2.0)) / (2.0 * pow(t.sigma, 2.0)));
        }
        let v = amp * t.contrast * sin(2.0 * pi * (t.sf * pos.x - t.phase));
        return vec4<f32>(t.color[0] + v, t.color[1] + v, t.color[2] + v, t.color[3]);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}