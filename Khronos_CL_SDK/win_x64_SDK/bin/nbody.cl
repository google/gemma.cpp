#define G 1.0f
//6.67384e-11f

#define epsilon_sq 1.f

float3 calculate_force(
    float3 first_pos,
    float3 second_pos,
    float first_mass,
    float second_mass)
{
    float3 d = first_pos - second_pos;
    float  q = sqrt(dot(d, d) + epsilon_sq);
    return -G * first_mass * second_mass * d / ( q*q*q );
}

kernel void nbody(global const float4* pos_mass_front,
                  global float4* pos_mass_back,
                  global float4* velocity,
                  uint particle_count,
                  float dt)
{
    size_t gid = get_global_id(0);
    float3 my_pos = pos_mass_front[gid].xyz;
    float my_mass = pos_mass_front[gid].w;
    float3 my_vel = velocity[gid].xyz;

    float3 acc = (float3)0.f;
    for (uint i = 0; i < particle_count; i++)
    {
        acc += calculate_force(
            my_pos,
            pos_mass_front[i].xyz,
            my_mass,
            pos_mass_front[i].w
        );
    }

    // updated position and velocity
    my_pos += my_vel * dt + acc * 0.5f * dt * dt;
    my_vel += acc * dt;

    // write to global memory
    pos_mass_back[gid] = (float4)(my_pos, my_mass);
    velocity[gid] = (float4)(my_vel, 0.f);
}
