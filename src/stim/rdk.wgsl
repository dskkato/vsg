// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) position: vec2<f32>,
    @location(2) color: vec3<f32>,
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
    out.color = model.color;
    return out;
}

struct RdkParams {
    diameter: f32,
};

@group(1) @binding(0)
var<uniform> t: RdkParams;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ctr = vec2<f32>(0.0, 0.0);

    if (distance(in.position, ctr) < t.diameter) {
        return vec4<f32>(in.color, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}