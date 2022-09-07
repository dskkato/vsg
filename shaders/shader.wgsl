// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) position: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
}

struct GratingParams {
    ctr: vec2<f32>,
};
@group(1) @binding(0)
var<uniform> params: GratingParams;

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
    out.clip_position = model_matrix * vec4<f32>(model.position, 1.0);
    out.position = model.position - vec3<f32>(params.ctr, 0.0);
    out.color = model.color;
    return out;
}

@group(0) @binding(0)
var<uniform> t: u32;

let pi = 3.14159;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if distance(in.position, vec3<f32>(0.0, 0.0, 0.0)) < 0.5 {
        let pos = in.position;
        let amp = (sin(2.0*pi*(10.0*pos.x - f32(t) / 60.0)) + 1.0) / 2.0;
        return vec4<f32>(amp, amp, amp, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.2);
    }
}