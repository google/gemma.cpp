#include "STFT.h"

std::vector<std::pair<std::string, std::string>>
OPENCL_ACC::get_platform_device_list()
{
    std::vector<std::pair<std::string, std::string>> pla_dev;
    std::vector<Platform> pvec;
    Platform::get(&pvec);
    for (int i = 0; i < pvec.size(); ++i) {
        std::string pname;
        std::vector<Device> dvec;
        pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
        pvec[i].getDevices(CL_DEVICE_TYPE_ALL, &dvec);
        for (int j = 0; j < dvec.size(); ++j) {
            std::pair<std::string, std::string> dpair;
            std::string dname;
            dvec[j].getInfo(CL_DEVICE_NAME, &dname);
            dpair.first = pname;
            dpair.second = dname;
            pla_dev.push_back(dpair);
        }
    }
    return pla_dev;
}
void
OPENCL_ACC::set_platform_device(const std::string& platform, const std::string& device)
{
    std::vector<Platform> pvec;
    Platform::get(&pvec);
    for (int i = 0; i < pvec.size(); ++i) {
        std::string pname;
        pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
        if (pname == platform) {
            this->PF = pvec[i];
            break;
        }
    }
    std::vector<Device> dvec;
    this->PF.getDevices(CL_DEVICE_TYPE_ALL, &dvec);
    for (int i = 0; i < dvec.size(); ++i) {
        std::string dname;
        dvec[i].getInfo(CL_DEVICE_NAME, &dname);
        if (dname == device) {
            this->DV = dvec[i];
            break;
        }
    }
    this->CT = Context(this->DV);
    this->ready = true;
}


OPENCL_ACC::OPENCL_ACC(const std::string& platform, const std::string& device)
{
    CLS = new CL_INSIDE();
    set_platform_device(platform, device);
}
OPENCL_ACC::OPENCL_ACC() {
    CLS = new CL_INSIDE();

}
OPENCL_ACC::~OPENCL_ACC() {
    delete CLS;
}





cl_float2*
OPENCL_ACC::overlap_and_extend_for_STFT(float* data_origin, const ma_uint64& origin_length, const ma_uint64& overlaped_length, const int& radix_2_size, const int& overlap_frame, const int& both_side_z_padding_size)
{
    int powed_size = pow(2, radix_2_size);
    cl_int2 tempint;
    tempint.x = origin_length / powed_size;
    tempint.y = origin_length % powed_size;
    cl_float2* overlaped = nullptr;
    gpgpu_facade<float, cl_float2>(
        CLS->overlap,
        data_origin,
        origin_length,
        overlaped,
        overlaped_length,
        overlaped_length,
        powed_size,
        overlap_frame,
        tempint,
        both_side_z_padding_size
    );//overlaped is alloced with new

    return overlaped;


}
void 
OPENCL_ACC::window_STFT(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_size)
{
    gpgpu_facade<cl_float2,cl_float2>(
        CLS->window_function,
        overlap_array,
        frame_length,
        overlap_array,
        frame_length,
        frame_length,
        window_radix_size
    );
}

void 
OPENCL_ACC::bit_reverse_STFT(cl_float2* data_array, const ma_uint64& frame_length, const int& window_radix_size)
{
    gpgpu_facade<cl_float2, cl_float2>(
        CLS->bit_reverse_STFT,
        data_array,
        frame_length,
        data_array,
        frame_length,
        frame_length,
        window_radix_size
    );
}

void
OPENCL_ACC::butterfly_STFT(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_2_size)
{
    int powed_size = pow(2, window_radix_2_size);
    Kernel KN = cl_facade::create_kernel(CLS->butterfly_STFT, "entry_point", CT, DV);
    CommandQueue CQ = clboost::make_cq(CT, DV);

    for (int stage = 0; stage < window_radix_2_size; ++stage) {

        Buffer dat_in = clboost::make_r_buf<cl_float2>(CT, frame_length, overlap_array);
        Buffer dat_out = clboost::make_w_buf<cl_float2>(CT, frame_length);
        clboost::set_args(KN, dat_in, dat_out, window_radix_2_size, stage);
        clboost::enq_q(CQ, KN, frame_length / 2);
        clboost::q_read<cl_float2>(CQ, dat_out, false, frame_length, overlap_array);
    }
}
float*
OPENCL_ACC::power_them(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_size)
{
    int powed_size = pow(2, window_radix_size);
    ma_uint64 only_usables = frame_length / 2;
    float* powered_data = nullptr;
    gpgpu_facade<cl_float2, float>(
        CLS->to_power,
        overlap_array,
        frame_length,
        powered_data,
        only_usables,
        only_usables,
        powed_size
    );
    delete[] overlap_array;
    return powered_data;

}
cl_float3*
OPENCL_ACC::three_bander(float* powered_STFT, const int& window_radix_size, int& low, int& mid, int& high, const int& quot)
{
    int powed_half = pow(2, window_radix_size-1);
    ma_uint64 only_usables = quot * powed_half;

    cl_float3* three_band_out = nullptr;

    //int FFT_expressible_HZ_range = DEFAULT_SAMPLERATE / 2;
    double freq_jump_size = ((double)DEFAULT_SAMPLERATE) / (double)(powed_half*2);

    int low_mid = (int)(((double)EQ_LOW_MID) / freq_jump_size)+1;
    int mid_high = (int)(((double)EQ_MID_HIGH) / freq_jump_size)+1;
    int high_toohigh = (int)(((double)EQ_HIGH_TOO_HIGH) / freq_jump_size) + 1;
    //11, 107-11, 512-107
    low = low_mid;
    int low_padded_size = to_big_radix_2(low);

    mid = mid_high - low_mid;
    int mid_padded_size = to_big_radix_2(mid);

    high = high_toohigh - mid_high;
    int high_padded_size = to_big_radix_2(high);

    float* lows = nullptr;
    float* mids = nullptr;
    float* highs = nullptr;
    gpgpu_facade<float, float>(
        CLS->split_low_band,
        powered_STFT,
        only_usables,
        lows,
        quot * low_padded_size,
        quot * low_padded_size,
        window_radix_size-1,
        low_mid,
        low_padded_size
    );
    gpgpu_facade<float, float>(
        CLS->split_mid_band,
        powered_STFT,
        only_usables,
        mids,
        quot * (mid_padded_size),
        quot * (mid_padded_size),
        window_radix_size - 1,
        low_mid,
        mid_high,
        mid_padded_size
    );
    gpgpu_facade<float, float>(
        CLS->split_high_band,
        powered_STFT,
        only_usables,
        highs,
        quot * (high_padded_size) ,
        quot * (high_padded_size),
        window_radix_size-1,
        mid_high,
        high_padded_size
    );
    three_band_out =
    three_divide_and_conquer(
        lows,
        low_padded_size,
        mids,
        mid_padded_size,
        highs,
        high_padded_size,
        quot
    );

    /*gpgpu_facade<float , cl_float3>(
        CLS->to_three_band,
        powered_STFT,
        only_usables/2,
        three_band_out,
        quot,
        quot,
        powed_size,
        low_mid,
        mid_high
    );*/
    delete[] powered_STFT;
    return three_band_out;


    //Kernel tbandKN = cl_facade::create_kernel(CLS->to_three_band, "entry_point", CT, DV);
    //
    //Buffer tband_in = clboost::make_r_buf<float>(CT, only_usables, powered_STFT);
    //Buffer tband_out = clboost::make_w_buf<cl_float3>(CT, quot);
    //
    //double freq_jump_size = ((double)DEFAULT_SAMPLERATE/2.0) / (double)(powed_size / 2);
    //
    //int low_mid = (int)round(((double)EQ_LOW_MID) / freq_jump_size);
    //int mid_high = (int)round(((double)EQ_MID_HIGH) / freq_jump_size);
    //clboost::set_args(tbandKN, tband_in, tband_out, powed_size/2, low_mid, mid_high);
    //clboost::enq_q(CQ, tbandKN, quot);
    //cl_float3 *t_band_out = new cl_float3[quot];
    //clboost::q_read<cl_float3>(CQ, tband_out, true, quot, t_band_out);
    //delete[] powered_STFT;
    //return t_band_out;
}
void 
OPENCL_ACC::thread_worker(Kernel& KN, CommandQueue& CQ, float*& data, const int& number_of_owner, const int& range)
{
    int number_of_calc = (int)log2(range);
    assert(range % 2 == 0);

    int accu_range = range;
    for (int i = 0; i < number_of_calc; ++i) {
        Buffer data_in = clboost::make_r_buf(CT, range * number_of_owner, data);
        Buffer data_out = clboost::make_w_buf<float>(CT, range * number_of_owner);
        clboost::set_args(KN, data_in, data_out);
        accu_range /= 2;
        clboost::enq_q(CQ, KN, accu_range*number_of_owner);
        clboost::q_read<float>(CQ, data_out, true,range * number_of_owner, data);
    }
}


cl_float3*
OPENCL_ACC::three_divide_and_conquer(
    float*& low, const int& low_range,
    float*& mid, const int& mid_range,
    float*& high, const int& high_range,
    const int& number_of_owner)
{
    Kernel lowKN = cl_facade::create_kernel(CLS->DaC, "entry_point", CT, DV);
    Kernel midKN = cl_facade::create_kernel(CLS->DaC, "entry_point", CT, DV);
    Kernel highKN = cl_facade::create_kernel(CLS->DaC, "entry_point", CT, DV);
    CommandQueue CQ = clboost::make_cq(CT, DV);
    std::thread lthread = std::thread([&]() {this->thread_worker(lowKN, CQ, low, number_of_owner, low_range); });
    std::thread mthread = std::thread([&]() {this->thread_worker(midKN, CQ, mid, number_of_owner, mid_range); });
    std::thread hthread = std::thread([&]() {this->thread_worker(highKN, CQ, high, number_of_owner, high_range); });

    lthread.join();
    mthread.join();
    hthread.join();
    Kernel finalKN = cl_facade::create_kernel(CLS->integrate_DaC, "entry_point", CT, DV);
    Buffer low_b = clboost::make_r_buf(CT, low_range * number_of_owner, low);
    Buffer mid_b = clboost::make_r_buf(CT, mid_range * number_of_owner, mid);
    Buffer high_b = clboost::make_r_buf(CT, high_range * number_of_owner, high);
    Buffer out_b = clboost::make_w_buf<cl_float3>(CT, number_of_owner);
    clboost::set_args(finalKN, low_b, mid_b, high_b, out_b);
    clboost::enq_q(CQ, finalKN, number_of_owner);
    cl_float3* three_band_out = new cl_float3[number_of_owner];
    clboost::q_read<cl_float3>(CQ, out_b, true, number_of_owner, three_band_out);
    delete[] low;
    delete[] mid;
    delete[] high;
    return three_band_out;
}










float*
OPENCL_ACC::cl_STFT(float* full_frame, const ma_uint64& full_length, const int& window_radix_2, const double& overlap_ratio, const int& front_side_z_padding_size, int& number_of_index)
{

    int powed_radix = pow(2, (int)window_radix_2);
    int fft_quotient = full_length / (int)((double)powed_radix * (1.0 - overlap_ratio));
    ma_uint64 overlaped_full_frame = fft_quotient * powed_radix;
    cl_float2* overlap_out = overlap_and_extend_for_STFT(full_frame, full_length, overlaped_full_frame, window_radix_2, (int)((double)powed_radix * (1.0 - overlap_ratio)), front_side_z_padding_size);

    window_STFT(overlap_out, overlaped_full_frame, window_radix_2);

    bit_reverse_STFT(overlap_out, overlaped_full_frame,window_radix_2);

    

    butterfly_STFT(overlap_out, overlaped_full_frame, window_radix_2);
    number_of_index = fft_quotient;
    return power_them(overlap_out, overlaped_full_frame, window_radix_2);

    /*cl_float3* three_band_out = three_bander(powered, overlaped_full_frame, window_radix_2, fft_quotient);

    number_of_index = fft_quotient;
    return three_band_out;*/
}

//
//void
//OPENCL_ACC::cl_fft(float* data_array, const int& data_length_radix_2)
//{
//    cl_float2 *complexed = bit_reverse(data_array, data_length_radix_2);
//    butterfly_stage_radix_2(complexed, data_length_radix_2,data_array);
//
//
//}









cl_float2*
OPENCL_ACC::bit_reverse(float* data_array, const int& data_length_radix_2)
{
    int powed_size = pow(2, data_length_radix_2);
    Kernel KN = cl_facade::create_kernel(CLS->bit_reverse, "entry_point", CT, DV);
    CommandQueue CQ = clboost::make_cq(CT, DV);
    Buffer dat_in = clboost::make_r_buf<float>(CT, powed_size, data_array);
    Buffer dat_out = clboost::make_w_buf<cl_float2>(CT, powed_size);
    clboost::set_args(KN, dat_in, dat_out, data_length_radix_2);
    clboost::enq_q(CQ, KN, powed_size);
    cl_float2* cfloat = new cl_float2[powed_size];
    clboost::q_read<cl_float2>(CQ, dat_out, true, powed_size, cfloat);
    return cfloat;
}


void 
OPENCL_ACC::butterfly_stage_radix_2(cl_float2* data, const int& data_length_radix_2, float* data_out)
{

    int powed_size = pow(2, data_length_radix_2);
    Kernel KN = cl_facade::create_kernel(CLS->butterfly_stage, "entry_point", CT, DV);
    CommandQueue CQ = clboost::make_cq(CT, DV);
    for (int stage = 0; stage < data_length_radix_2; ++stage) {
        Buffer dat_in = clboost::make_r_buf<cl_float2>(CT, powed_size, data);
        Buffer dat_out = clboost::make_w_buf<cl_float2>(CT, powed_size);
        clboost::set_args(KN, dat_in, dat_out, data_length_radix_2, stage);
        clboost::enq_q(CQ, KN, powed_size/2);
        clboost::q_read<cl_float2>(CQ, dat_out, true, powed_size, data);        
    }

    Kernel final_KN = cl_facade::create_kernel(CLS->to_power, "entry_point", CT, DV);
    Buffer fin_in = clboost::make_r_buf<cl_float2>(CT, powed_size, data);
    Buffer fin_out = clboost::make_w_buf<float>(CT, powed_size);
    clboost::set_args(final_KN, fin_in, fin_out);
    clboost::enq_q(CQ, final_KN, powed_size);
    clboost::q_read<float>(CQ, fin_out, true, powed_size, data_out);
    delete[] data;
}


void
OPENCL_ACC::to_dbfs(cl_float3* data, const int& window_radix_size, const int& low, const int& mid, const int& high, const int& quot )
{
    int wos = pow(2, window_radix_size);
    gpgpu_facade<cl_float3, cl_float3>(
        CLS->to_dbfs,
        data,
        quot,
        data,
        quot,
        quot,
        wos,
        low,
        mid,
        high
    );
    
}

//
//
////
//#include "miniaudio.h"
////
//#include <iostream>
//int main() {
//        OPENCL_ACC oa = OPENCL_ACC();
//    std::vector<std::pair<std::string, std::string>> list = oa.get_platform_device_list();
//    /*for(int i = 0; i < list.size(); ++i) {
//        std::cout << list[i].first << "--" << list[i].second << std::endl;
//    }*/
//    oa.set_platform_device(
//        "NVIDIA CUDA", "NVIDIA GeForce GTX 1660 Ti with Max-Q Design");//임시로 하드코딩
//
//
//    //oa.STFT_TESTER();
//    int radix = 15;
//
//    ma_decoder_config deconf = ma_decoder_config_init(ma_format_f32, 1, 48000);
//    ma_decoder dec;
//    ma_decoder_init_file("E:/Word Of Old.wav", &deconf, &dec);
//    ma_uint64 length;
//    ma_decoder_get_length_in_pcm_frames(&dec, &length);
//    length /= 5;
//    float* full_frame = new float[length];
//    ma_uint64 check;
//    ma_decoder_read_pcm_frames(&dec, full_frame, length, &check);
//    ASSERT_EQ(check, length);
//    int quot = 0;
//    int low = 0;
//    int mid = 0;
//    int high = 0;
//    float* stft_output = oa.cl_STFT(full_frame, length, radix, 0.9, 16384, quot);
//    cl_float3* tband = oa.three_bander(stft_output, radix, low, mid, high, quot);
//    oa.to_dbfs(tband, radix, low, mid, high, quot);
//    for (int i = 0; i < quot; ++i) {
//        std::cout << tband[i].x << "," << tband[i].y << "," << tband[i].z << std::endl;
//    }
//    getchar();
//
//
//
//    return 0;
//}






























//
//#define TD_OVER 10000
//
//float*
//TD_overlap()
//{
//    float* test_arr = new float[TD_OVER];
//    for (int i = 0; i < TD_OVER; ++i) {
//        test_arr[i] = i;
//    }
//    return test_arr;
//}
//
//bool
//TO_overlap_tester(cl_float2* test_set, float* origin, int quot, ma_uint64 out_size)
//{
//    int index = 0;
//    for (int i = 0; i < quot; ++i) {
//        for (int j = 0; j < 1024; ++j) {
//            if (i * 512 + j >= TD_OVER) {
//                goto TO_GOOD_BREAK;
//            }
//            if (origin[i * 512 + j] != test_set[index].x||test_set[index].y!=0) {
//                
//                return false;
//            }
//            ++index;
//        }
//    }
//    TO_GOOD_BREAK:
//    delete[] test_set;
//    delete[] origin;
//    return true;
//}
//
//
//cl_float2*
//TD_hamming() {
//    cl_float2* test_data = new cl_float2[1024*10];
//    for (int i = 0; i < 10240; ++i) {
//        test_data[i].x = 0;
//        test_data[i].y = 0;
//    }
//    return test_data;
//}
//
//
//
//bool
//TO_hamming_test(cl_float2* out_data)
//{
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 1024; ++j) {
//            float hamming = 0.54f - 0.46f * cosf(2.0f * CL_M_PI * (float)j / 1023.0f);
//            if (std::abs(hamming - out_data[i * 1024 + j].x)>=0.01 || 0 != out_data[i * 1024 + j].y) {
//                return false;
//            }
//        }
//    }
//    delete[] out_data;
//    return true;
//}
//
//
//cl_float2*
//TD_bit_rev()
//{
//    cl_float2* test_data = new cl_float2[10240];
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 1024; ++j) {
//            test_data[i * 1024 + j].x = i;
//            test_data[i * 1024 + j].y = 0;
//        }
//    }
//    return test_data;
//}
//
//int reverseBits(int num, int radix_2_data) {
//    int reversed = 0;
//    for (int i = 0; i < radix_2_data; ++i) {
//        reversed = (reversed << 1) | (num & 1);
//        num >>= 1;
//    }
//    return reversed;
//}
//
//
//
//
//bool
//TO_rev_test(cl_float2* test_data)
//{
//    float origin[1024];
//    for (int i = 0; i < 1024; ++i) {
//        origin[i] = reverseBits(i,10);
//    }
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 1024; ++j) {
//            if (test_data[i * 1024 + j].x != origin[j] || test_data[i * 1024 + j].y != 0) {
//                return false;
//            }
//        }
//    }
//    return true;
//}
//
//cl_float2*
//TD_power_test()
//{
//    cl_float2* test_data = new cl_float2[1024 * 10];
//    for (int i = 0; i < 10240; ++i) {
//        test_data[i].x = i%1024;
//        test_data[i].y = i%1024;
//    }
//    return test_data;
//}
//
//bool
//TO_power(float* power_Data)
//{
//    for (int i = 0; i < 10240/2; ++i) {
//        int j = i % 512;
//        float sq = sqrtf(2.0f * powf(j, 2));
//        if (std::abs(power_Data[i] - sq) >= 0.0001) {
//            return false;
//        }
//    }
//    return true;
//}
//
//
//
//
//
//float*
//TD_three_bander()
//{
//    float* test_set = new float[5120];
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 512; ++j) {
//            test_set[i * 512 + j] = j;
//        }
//    }
//    return test_set;
//}
//
//bool
//TO_test_three_bander(cl_float3* out_data)
//{//11, 107
//    int low_side = 0;
//    int mid_side = 0;
//    int high_side = 0;
//    for (int i = 0; i < 11; ++i) {
//        low_side += i;
//    }
//    for (int i = 11; i < 107; ++i) {
//        mid_side += i;
//    }
//    for (int i = 107; i < 512; ++i) {
//        high_side += i;
//    }
//
//    for (int i = 0; i < 10; ++i) {
//        if (out_data[i].x != low_side || out_data[i].y != mid_side || out_data[i].z != high_side) {
//            return false;
//        }
//    }
//    return true;
//}
//#include <random>
//
//float* TD_butterfly_tester()
//{
//    float* test_set = new float[10240];
//    std::random_device rd;
//    std::mt19937 gen(rd());
//
//    // -1.0에서 1.0까지의 실수 생성을 위한 분포 설정
//    std::uniform_real_distribution<float> dis(-1.0, 1.0);
//    for (int i = 0; i < 10240; ++i) {
//        test_set[i] = dis(gen);
//    }
//    return test_set;
//}
//
//
//#include <complex>
//// DFT 계산 함수
//std::vector<std::complex<float>> 
//DFT(float* signal, const int& data_size) {
//    int N = data_size;
//    std::vector<std::complex<float>> spectrum(N);
//
//    for (int k = 0; k < N; ++k) {
//        std::complex<float> sum(0.0, 0.0);
//        for (int n = 0; n < N; ++n) {
//            sum += signal[n] * std::exp(std::complex<float>(0, -2 * CL_M_PI * k * n / N));
//        }
//        spectrum[k] = sum;
//    }
//    return spectrum;
//}
//
//std::vector<std::complex<float>>
//DFT_STFT(float* signal, const int& data_size, const int& window_size_radix_2) {
//    int powed = pow(2, window_size_radix_2);
//    std::vector<std::complex<float>> spectrum;
//    for (int i = 0; i < data_size/powed; ++i) {
//        float* windowed_signal = new float[powed];
//        for (int j = 0; j < powed; ++j) {
//            windowed_signal[j] = signal[i * powed + j];
//        }
//        std::vector<std::complex<float>> window_spectrum(powed);
//        window_spectrum = DFT(windowed_signal, powed);
//        delete[] windowed_signal;
//        for (int z = 0; z < window_spectrum.size(); ++z) {
//            spectrum.push_back(window_spectrum[z]);
//        }
//    }
//    return spectrum;
//}
//
//
//
//bool
//TO_DFT_TESTER(cl_float2* test_target, std::vector<std::complex<float>> DFT_real, const int& window_size)
//{
//    for (int i = 0; i < window_size; ++i) {
//        float D_real = DFT_real[i].real();
//        float D_imag = DFT_real[i].imag();
//        if (std::abs(test_target[i].x -D_real)>0.1 || std::abs(test_target[i].y - D_imag) > 0.1) {
//            std::cout << "diffrence detect" << std::abs(test_target[i].x - D_real) << "imag-" << std::abs(test_target[i].y - D_imag) << std::endl;
//        }
//    }
//    return true;
//}
//
//
//
//
//void
//OPENCL_ACC::STFT_TESTER()
//{
//
//    //test1
//    int powed_radix = pow(2, 10);
//    int fft_quotient = TD_OVER / (powed_radix / 2);
//    ma_uint64 overlaped_full_frame = fft_quotient*powed_radix;
//    
//    float* TD_SET_OVER = TD_overlap();
//    cl_float2* TO_OVER = overlap_and_extend_for_STFT(TD_SET_OVER,TD_OVER,overlaped_full_frame,10,powed_radix/2);
//    TO_overlap_tester(TO_OVER, TD_SET_OVER,fft_quotient,overlaped_full_frame);
//
//
//    cl_float2 *hamming_test_data = TD_hamming();
//
//    hamming_window_STFT(hamming_test_data, 10240,10);
//
//    TO_hamming_test(hamming_test_data);
//
//    cl_float2* bit_reverse_test_data = TD_bit_rev();
//
//    bit_reverse_STFT(bit_reverse_test_data,10240,10);
//
//    delete[] bit_reverse_test_data;
//    cl_float2* power_test_data = TD_power_test();
//
//
//    float* powered_data = power_them(power_test_data, 10240, 10);
//    TO_power(powered_data);
//    delete[] powered_data;
//
//
//    float* three_bander_data = TD_three_bander();
//    cl_float3* out_dat = three_bander(three_bander_data, 10240, 10, 10);
//
//    TO_test_three_bander(out_dat);
//    delete[] out_dat;
//    float* random_float_data = TD_butterfly_tester();
//    cl_float2* butterfly_data = new cl_float2[10240];
//    for (int i = 0; i < 10240; ++i) {
//        butterfly_data[i].x = random_float_data[i];
//        butterfly_data[i].y = 0;
//    }
//
//    bit_reverse_STFT(butterfly_data, 10240, 10);
//
//
//    std::vector<std::complex<float>> DFT_out = DFT_STFT(random_float_data, 10240, 10);
//    butterfly_STFT(butterfly_data, 10240, 10);
//    TO_DFT_TESTER(butterfly_data, DFT_out, 10240);
//
//
//    /*
//
//    butterfly_STFT();*/
//
//
//
//}

