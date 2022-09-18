// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) position: vec2<f32>,
    @location(2) sf: f32, // spatial frequency
    @location(4) tf: f32, // temporal frequency
    @location(5) phase: f32,
    @location(6) contrast: f32,
    @location(7) tick: f32,
    @location(8) diameter: f32,
    @location(9) sigma: f32,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) sf: f32, // spatial frequency
    @location(10) tf: f32, // temporal frequency
    @location(11) phase: f32,
    @location(12) contrast: f32,
    @location(13) tick: f32,
    @location(14) diameter: f32,
    @location(15) sigma: f32,
    @location(16) padding: f32,
    @location(17) color: vec4<f32>,
}

struct ViewUniform {
    proj: mat4x4<f32>,
};
@group(0) @binding(0)
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
    out.sf = instance.sf;
    out.tf = instance.tf;
    out.phase = instance.phase;
    out.contrast = instance.contrast;
    out.tick = instance.tick;
    out.diameter = instance.diameter;
    out.sigma = instance.sigma;
    out.color = instance.color;
    return out;
}

let pi = 3.1415926535897;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ctr = vec2<f32>(0.0, 0.0);

    if (distance(in.position, ctr) < in.diameter) {
        let pos = in.position.xy;
        var amp: f32;
        if (in.sigma <= 0.0) {
            amp = 1.0;
        } else {
            amp = exp(-(pow(pos.x, 2.0) + pow(pos.y, 2.0)) / (2.0 * pow(in.sigma, 2.0)));
        }
        let v = amp * in.contrast * sin(2.0 * pi * (in.sf * pos.x - in.phase));
        return vec4<f32>(in.color[0] + v, in.color[1] + v, in.color[2] + v, in.color[3]);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}