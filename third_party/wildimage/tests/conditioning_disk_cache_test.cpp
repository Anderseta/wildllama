#include "wildimage_conditioning_cache.h"
#include "core/tensor.hpp"
#include <fstream>
#include <iostream>

using namespace wildimage_conditioning;

struct RefParams {
    bool pass_to_vlm=false, pass_to_dit=true;
    int ref_index_mode=0;
    bool force_ref_timestep_zero=false, resize_before_vae=true;
    int vae_input_max_pixels=-1, vlm_resize_mode=2, vlm_min_size=-1, vlm_max_size=-1;
    bool resize_vae_to_target=false, resize_single_ref_to_target=false, vae_size_buckets=false, vae_center_crop=false, vlm_use_vae_input=false;
    int ref_image_size=0;
};
struct Params {
    std::string text="an exact prompt";
    int clip_skip=-1, width=512, height=768;
    bool zero_out_masked=false;
    const std::vector<sd::Tensor<float>>* ref_images=nullptr;
    const std::vector<int>* minimax_h3_references=nullptr;
    RefParams ref_image_params;
};
struct Condition {
    sd::Tensor<float> c_crossattn,c_vector,c_concat,c_t5_weights;
    sd::Tensor<int32_t> c_t5_ids,c_input_ids,c_position_ids,c_token_types,c_vinput_mask;
    std::vector<std::pair<int,sd::Tensor<float>>> c_image_embeds;
    std::vector<sd::Tensor<float>> c_ref_images,c_ref_audios,extra_c_crossattns;
    std::vector<int> c_reference_blocks;
    bool empty() const {return c_crossattn.empty()&&c_vector.empty()&&c_concat.empty()&&c_t5_ids.empty()&&c_t5_weights.empty()&&
        c_input_ids.empty()&&c_position_ids.empty()&&c_token_types.empty()&&c_vinput_mask.empty()&&c_image_embeds.empty()&&
        c_ref_images.empty()&&c_ref_audios.empty()&&extra_c_crossattns.empty();}
};
static void check(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
static std::string key(const std::string& value) {return hex(sha256(value.data(),value.size()));}
static uint64_t owned_bytes(const fs::path& directory) {
    uint64_t total=0;
    for(const auto& file:fs::directory_iterator(directory)) if(file.path().filename().string().find("wic1-")==0&&safe_regular(file.path())) total+=file.file_size();
    return total;
}

int main(int argc,char** argv) {
    try {
        check(argc==2,"pass isolated G: test directory");
        const fs::path root=fs::u8path(argv[1]);
        check(root.is_absolute(),"absolute test directory required");
        fs::create_directories(root);
        check(key("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","NIST empty SHA256");
        check(key("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","NIST abc SHA256");
        check(key("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1","NIST multiblock SHA256");
        SHA256 million; const std::string chunk(1000,'a'); for(int i=0;i<1000;++i) million.update(chunk.data(),chunk.size());
        check(hex(million.finish())=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","NIST million-a SHA256");
        Config config{root/"exact",key("model/profile namespace"),1u<<20};
        Cache<Condition,Params> cache(config); Params params;
        Condition expected;
        const std::vector<float> values={0.0f,-0.0f,1.25f,-17.0f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()};
        auto tensor=sd::Tensor<float>({2,3},values);
        expected.c_crossattn=tensor; expected.c_vector=tensor; expected.c_concat=tensor; expected.c_t5_weights=tensor;
        auto integers=sd::Tensor<int32_t>({4},{INT32_MIN,-1,0,INT32_MAX});
        expected.c_t5_ids=integers; expected.c_input_ids=integers; expected.c_position_ids=integers;
        expected.c_token_types=integers; expected.c_vinput_mask=integers;
        expected.c_image_embeds.emplace_back(-3,tensor);
        expected.c_ref_images={tensor,sd::Tensor<float>({0,4})}; expected.c_ref_audios={tensor}; expected.extra_c_crossattns={tensor,tensor};
        const auto original=encode(expected,1u<<20);
        const auto original_key=cache.key(params,1);
        check(cache.save(original_key,expected),"save full condition");
        Condition actual; check(cache.load(original_key,actual),"load full condition");
        check(encode(actual,1u<<20)==original,"round trip ALL tensor bits/shapes/fields");
        auto scalar=sd::Tensor<float>({1},{-0.0f}); scalar.squeeze_(); expected.c_vector=scalar;
        const auto scalar_wire=encode(expected,1u<<20); Condition scalar_back;
        check(decode(scalar_wire,scalar_back)&&encode(scalar_back,1u<<20)==scalar_wire,"zero-rank scalar and signed zero exact");
        auto miss=[&](const Params& changed){check(cache.key(changed,1)!=original_key,"changed actual parameter must miss");};
        auto p=params; p.text+="!"; miss(p); p=params;p.clip_skip=2;miss(p);p=params;p.width++;miss(p);p=params;p.height++;miss(p);
        p=params;p.zero_out_masked=true;miss(p);
#define CHANGE_REF(field,value) p=params;p.ref_image_params.field=value;miss(p)
        CHANGE_REF(pass_to_vlm,true);CHANGE_REF(pass_to_dit,false);CHANGE_REF(ref_index_mode,2);
        CHANGE_REF(force_ref_timestep_zero,true);CHANGE_REF(resize_before_vae,false);CHANGE_REF(vae_input_max_pixels,1024);
        CHANGE_REF(vlm_resize_mode,1);CHANGE_REF(vlm_min_size,256);CHANGE_REF(vlm_max_size,512);
        CHANGE_REF(resize_vae_to_target,true);CHANGE_REF(resize_single_ref_to_target,true);CHANGE_REF(vae_size_buckets,true);
        CHANGE_REF(vae_center_crop,true);CHANGE_REF(vlm_use_vae_input,true);CHANGE_REF(ref_image_size,1);
#undef CHANGE_REF
        check(cache.key(params,2)!=original_key,"resolved clip skip hashed");
        Cache<Condition,Params> another_namespace({root/"exact",key("changed encoder"),1u<<20});
        check(another_namespace.key(params,1)!=original_key,"namespace model stamp hashed");
        std::vector<sd::Tensor<float>> refs{tensor};p=params;p.ref_images=&refs;const auto ref_key=cache.key(p,1);
        check(ref_key!=original_key,"actual references hashed"); refs[0].values()[0]=-0.0f;
        check(cache.key(p,1)!=ref_key,"reference tensor bits hashed");
        refs[0].values()[0]=std::numeric_limits<float>::quiet_NaN();check(cache.key(p,1).empty(),"NaN reference bypass");
        std::vector<int> minimax;p=params;p.minimax_h3_references=&minimax;check(cache.key(p,1).empty(),"unsupported MiniMax input bypass");
        expected.c_reference_blocks={1};check(!cache.save(key("h3"),expected),"unsupported H3 output bypass");expected.c_reference_blocks.clear();
        expected.c_crossattn.values()[0]=std::numeric_limits<float>::infinity();check(!cache.save(key("nonfinite"),expected),"nonfinite output bypass");

        Store disk(config); Bytes bytes;
        const auto corrupt=disk.path(original_key);
        {std::fstream file(corrupt,std::ios::in|std::ios::out|std::ios::binary);file.seekp(-1,std::ios::end);char c=char(0xff);file.write(&c,1);}
        check(!cache.load(original_key,actual)&&!fs::exists(corrupt),"checksum corruption rejected and removed");
        check(disk.save(original_key,original),"resave after corruption");fs::resize_file(corrupt,81);
        check(!disk.load(original_key,bytes)&&!fs::exists(corrupt),"truncated entry rejected");
        Bytes malformed=original; for(size_t i=8;i<12;++i) malformed[i]=0xff;
        check(disk.save(original_key,malformed),"self-consistent malformed fixture");
        const auto before=encode(actual,1u<<20);
        check(!cache.load(original_key,actual)&&!fs::exists(corrupt)&&encode(actual,1u<<20)==before,"rank bound checked before tensor allocation, output unchanged");

        Config small{root/"lru",key("namespace"),400}; Store lru(small); const Bytes body(100,0x5a);
        const auto a=key("a"),b=key("b"),c=key("c");
        check(lru.save(a,body)&&lru.save(b,body),"two bounded entries");
        fs::last_write_time(lru.path(a),fs::file_time_type::clock::now()-std::chrono::seconds(2));
        fs::last_write_time(lru.path(b),fs::file_time_type::clock::now()-std::chrono::seconds(1));
        check(lru.load(a,bytes)&&bytes==body,"LRU hit");check(lru.save(c,body),"LRU third entry");
        check(fs::exists(lru.path(a))&&!fs::exists(lru.path(b))&&fs::exists(lru.path(c)),"LRU oldest evicted");
        check(owned_bytes(small.directory)<=small.budget,"strict total budget including envelope");
        small.budget=190;Store reduced(small);check(reduced.load(a,bytes)&&owned_bytes(small.directory)<=190,"reduce budget on GET, retain requested hit");
        small.budget=179;Store tiny(small);check(!tiny.load(a,bytes)&&owned_bytes(small.directory)==0,"entry larger than reduced budget removed");
        check(!tiny.save(c,body),"per entry cap includes envelope");
        const auto foreign=small.directory/"do-not-delete.txt";{std::ofstream file(foreign);file<<"preserved";}
        check(!tiny.load(a,bytes)&&fs::exists(foreign),"owned files only");
        Config disabled{root/"disabled",key("namespace"),0};Store off(disabled);
        check(!off.load(a,bytes)&&!off.save(a,body)&&!fs::exists(disabled.directory),"zero disables all filesystem effects");
        check(lru.save(a,body),"restore lock fixture");
        {Lock held;check(held.acquire(small.directory),"fixture lock acquired");check(!lru.load(a,bytes)&&!lru.save(b,body),"busy OS lock bypasses without waiting or writes");}
        const auto unicode=root/fs::u8path(u8"cache-ç-中");
        const auto stamp=key("environment namespace");
#ifdef _WIN32
        _putenv_s("WILDIMAGE_CONDITION_NAMESPACE",stamp.c_str());
        _putenv_s("WILDIMAGE_CONDITION_CACHE_BYTES","1024");
        check(SetEnvironmentVariableW(L"WILDIMAGE_CONDITION_CACHE_DIR",unicode.c_str())!=0,"set Unicode cache directory");
#else
        setenv("WILDIMAGE_CONDITION_NAMESPACE",stamp.c_str(),1);
        setenv("WILDIMAGE_CONDITION_CACHE_BYTES","1024",1);
        setenv("WILDIMAGE_CONDITION_CACHE_DIR",unicode.c_str(),1);
#endif
        const auto environment=Config::environment();
        check(environment.enabled()&&environment.directory==unicode,"native Unicode environment directory");
        Store wide(environment);check(wide.save(a,body)&&wide.load(a,bytes)&&bytes==body,"Unicode path roundtrip");
        std::cout<<"PASS SHA256 NIST; exact full-condition bits; complete key invalidation; finite/H3 guards; corruption/rank/truncation; strict LRU budgets; lock bypass; disabled mode\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<"\n";return 1;}
}
