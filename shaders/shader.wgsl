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
    out.position = model.position;
    out.color = model.color;
    return out;
}

struct GratingParams {
    sf: f32, // spatial frequency
    _tf: f32, // temporal frequency
    phase: f32,
    contrast: f32,
    _tick: f32,
    diameter: f32,
    sigma: f32,
    red: f32,
    blue: f32,
    green: f32,
};

@group(0) @binding(0)
var<uniform> t: GratingParams;

let pi = 3.14159;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if distance(in.position, vec3<f32>(0.0, 0.0, 0.0)) < t.diameter {
        if t.sigma <= 0.0 {
            let pos = in.position;
            let v = (t.contrast * sin(2.0*pi*(t.sf * pos.x - t.phase)) + 1.0) / 2.0;
            return vec4<f32>(v, v, v, 1.0);
        }
        else {
            let pos = in.position;
            let amp = exp(-(pow(pos.x, 2.0) + pow(pos.y, 2.0)) / (2.0 * pow(t.sigma, 2.0)));
            let v = amp * t.contrast * sin(2.0*pi*(t.sf * pos.x - t.phase));
            return vec4<f32>(t.red + v, t.green + v, t.blue + v, 1.0);
        }
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}