#ifndef WILDIMAGE_CONDITIONING_CACHE_H
#define WILDIMAGE_CONDITIONING_CACHE_H

#include "wildimage_conditioning_store.h"
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace wildimage_conditioning {

inline bool supported_host() {
    const uint16_t n=1;
    return *reinterpret_cast<const uint8_t*>(&n)==1 && sizeof(float)==4 && sizeof(int32_t)==4 && std::numeric_limits<float>::is_iec559;
}
inline void require(bool value) {if(!value) throw std::runtime_error("invalid conditioning cache entry");}

class HashWriter {
    SHA256 hash_;
    uint64_t used_=0;
public:
    size_t tensors=0;
    void bytes(const void* data,size_t size) {
        require(size<=entry_max_bytes && used_<=entry_max_bytes-size);
        used_+=size; hash_.update(data,size);
    }
    void u32(uint32_t value) {uint8_t p[4]; for(unsigned i=0;i<4;++i) p[i]=uint8_t(value>>(8*i)); bytes(p,4);}
    void u64(uint64_t value) {uint8_t p[8]; put64(p,value); bytes(p,8);}
    void text(const std::string& value) {require(value.size()<=(1u<<20)); u64(value.size()); bytes(value.data(),value.size());}
    std::string finish() const {return hex(hash_.finish());}
};

class Writer {
    size_t limit_;
public:
    Bytes data;
    size_t tensors=0;
    explicit Writer(size_t limit):limit_(limit){}
    void bytes(const void* source,size_t size) {
        require(size<=limit_ && data.size()<=limit_-size);
        if(size) {const auto* p=static_cast<const uint8_t*>(source); data.insert(data.end(),p,p+size);}
    }
    void u32(uint32_t value) {uint8_t p[4]; for(unsigned i=0;i<4;++i) p[i]=uint8_t(value>>(8*i)); bytes(p,4);}
    void u64(uint64_t value) {uint8_t p[8]; put64(p,value); bytes(p,8);}
};

template<class Tensor>
using TensorValue=typename std::decay<decltype(std::declval<Tensor>().values())>::type::value_type;

template<class Sink,class Tensor>
inline void write_tensor(Sink& sink,const Tensor& tensor) {
    using T=TensorValue<Tensor>;
    static_assert(std::is_same<T,float>::value||std::is_same<T,int32_t>::value,"conditioning dtype must be float32 or int32");
    require(++sink.tensors<=1024 && tensor.shape().size()<=8 && tensor.values().size()<=entry_max_bytes/4);
    uint64_t product=tensor.shape().empty()?0:1;
    for(int64_t dim:tensor.shape()) {
        require(dim>=0 && uint64_t(dim)<=INT32_MAX);
        if(!dim) product=0;
        else {require(product<=entry_max_bytes/4/uint64_t(dim)); product*=uint64_t(dim);}
    }
    const uint64_t count=tensor.values().size();
    require((tensor.shape().empty()&&count<=1)||(!tensor.shape().empty()&&count==product));
    if constexpr(std::is_same<T,float>::value) for(float value:tensor.values()) require(std::isfinite(value));
    sink.u32(std::is_same<T,float>::value?1:2); sink.u32(uint32_t(tensor.shape().size())); sink.u64(count);
    for(int64_t dim:tensor.shape()) sink.u64(uint64_t(dim));
    sink.bytes(tensor.data(),size_t(count)*4);
}

class Reader {
    const Bytes& data_;
    size_t at_=0;
public:
    size_t tensors=0;
    explicit Reader(const Bytes& data):data_(data){}
    const uint8_t* bytes(size_t count) {
        require(count<=data_.size()-at_);
        const auto* p=data_.data()+at_; at_+=count; return p;
    }
    uint32_t u32() {const auto* p=bytes(4); return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
    uint64_t u64() {return get64(bytes(8));}
    bool done() const {return at_==data_.size();}
    template<class Tensor>
    void tensor(Tensor& output,bool materialize) {
        using T=TensorValue<Tensor>;
        require(++tensors<=1024 && u32()==(std::is_same<T,float>::value?1u:2u));
        const uint32_t rank=u32(); const uint64_t count=u64();
        require(rank<=8 && count<=entry_max_bytes/4);
        std::array<int64_t,8> shape{};
        uint64_t product=rank?1:0;
        for(uint32_t i=0;i<rank;++i) {
            const uint64_t dim=u64(); require(dim<=INT32_MAX); shape[i]=int64_t(dim);
            if(!dim) product=0;
            else {require(product<=entry_max_bytes/4/dim); product*=dim;}
        }
        require((rank==0&&count<=1)||(rank!=0&&count==product));
        const auto* raw=bytes(size_t(count)*4);
        if(!materialize) {
            if constexpr(std::is_same<T,float>::value) for(uint64_t i=0;i<count;++i) {
                float value; std::memcpy(&value,raw+size_t(i)*4,4); require(std::isfinite(value));
            }
        } else {
            output=Tensor(std::vector<int64_t>(shape.begin(),shape.begin()+rank));
            output.values().resize(size_t(count));
            if(count) std::memcpy(output.data(),raw,size_t(count)*4);
        }
    }
};

template<class Sink,class Values>
inline void write_list(Sink& writer,const Values& values) {
    require(values.size()<=256); writer.u32(uint32_t(values.size()));
    for(const auto& tensor:values) write_tensor(writer,tensor);
}
template<class Values>
inline void read_list(Reader& reader,Values& values,bool materialize) {
    const uint32_t count=reader.u32(); require(count<=256);
    if(materialize) values.reserve(count);
    for(uint32_t i=0;i<count;++i) {
        typename Values::value_type tensor;
        reader.tensor(tensor,materialize);
        if(materialize) values.push_back(std::move(tensor));
    }
}

template<class Condition>
inline Bytes encode(const Condition& condition,size_t limit) {
    require(supported_host() && condition.c_reference_blocks.empty());
    Writer writer(limit); writer.u32(1);
    write_tensor(writer,condition.c_crossattn); write_tensor(writer,condition.c_vector); write_tensor(writer,condition.c_concat);
    write_tensor(writer,condition.c_t5_ids); write_tensor(writer,condition.c_t5_weights);
    write_tensor(writer,condition.c_input_ids); write_tensor(writer,condition.c_position_ids);
    write_tensor(writer,condition.c_token_types); write_tensor(writer,condition.c_vinput_mask);
    require(condition.c_image_embeds.size()<=256); writer.u32(uint32_t(condition.c_image_embeds.size()));
    for(const auto& item:condition.c_image_embeds) {writer.u32(uint32_t(item.first)); write_tensor(writer,item.second);}
    write_list(writer,condition.c_ref_images); write_list(writer,condition.c_ref_audios); write_list(writer,condition.extra_c_crossattns);
    return std::move(writer.data);
}

template<class Condition>
inline void read_condition(Reader& reader,Condition& condition,bool materialize) {
    require(reader.u32()==1);
    reader.tensor(condition.c_crossattn,materialize); reader.tensor(condition.c_vector,materialize); reader.tensor(condition.c_concat,materialize);
    reader.tensor(condition.c_t5_ids,materialize); reader.tensor(condition.c_t5_weights,materialize);
    reader.tensor(condition.c_input_ids,materialize); reader.tensor(condition.c_position_ids,materialize);
    reader.tensor(condition.c_token_types,materialize); reader.tensor(condition.c_vinput_mask,materialize);
    const uint32_t count=reader.u32(); require(count<=256);
    if(materialize) condition.c_image_embeds.reserve(count);
    for(uint32_t i=0;i<count;++i) {
        const uint32_t index=reader.u32();
        typename std::decay<decltype(condition.c_crossattn)>::type tensor;
        reader.tensor(tensor,materialize);
        if(materialize) {int32_t signed_index; std::memcpy(&signed_index,&index,4); condition.c_image_embeds.emplace_back(signed_index,std::move(tensor));}
    }
    read_list(reader,condition.c_ref_images,materialize); read_list(reader,condition.c_ref_audios,materialize);
    read_list(reader,condition.extra_c_crossattns,materialize);
    require(reader.done());
}

template<class Condition>
inline bool decode(const Bytes& bytes,Condition& output) noexcept {
    try {
        require(supported_host() && bytes.size()<=entry_max_bytes);
        // Validate all shapes/counts and finite data before allocating any tensor.
        Condition probe; Reader validation(bytes); read_condition(validation,probe,false);
        Condition condition; Reader materialization(bytes); read_condition(materialization,condition,true);
        output=std::move(condition); return true;
    } catch(...) {return false;}
}

template<class Condition,class Params>
class Cache {
    Store store_;
public:
    explicit Cache(Config config=Config::environment()):store_(std::move(config)){}
    std::string key(const Params& params,int resolved_clip_skip) const noexcept {
        try {
            if(!store_.config().enabled() || !supported_host() || params.minimax_h3_references!=nullptr) return {};
            HashWriter writer; writer.text("wildimage-conditioning-v1"); writer.text(store_.config().name_space);
            writer.text(params.text); writer.u32(uint32_t(params.clip_skip)); writer.u32(uint32_t(resolved_clip_skip));
            writer.u32(uint32_t(params.width)); writer.u32(uint32_t(params.height)); writer.u32(params.zero_out_masked);
            const auto& r=params.ref_image_params;
            writer.u32(r.pass_to_vlm); writer.u32(r.pass_to_dit); writer.u32(uint32_t(r.ref_index_mode));
            writer.u32(r.force_ref_timestep_zero); writer.u32(r.resize_before_vae); writer.u32(uint32_t(r.vae_input_max_pixels));
            writer.u32(uint32_t(r.vlm_resize_mode)); writer.u32(uint32_t(r.vlm_min_size)); writer.u32(uint32_t(r.vlm_max_size));
            writer.u32(r.resize_vae_to_target); writer.u32(r.resize_single_ref_to_target); writer.u32(r.vae_size_buckets);
            writer.u32(r.vae_center_crop); writer.u32(r.vlm_use_vae_input); writer.u32(uint32_t(r.ref_image_size));
            writer.u32(params.ref_images!=nullptr);
            if(params.ref_images!=nullptr) {
                require(params.ref_images->size()<=64); writer.u32(uint32_t(params.ref_images->size()));
                for(const auto& tensor:*params.ref_images) write_tensor(writer,tensor);
            }
            return writer.finish();
        } catch(...) {return {};}
    }
    bool load(const std::string& key,Condition& output) const noexcept {
        if(key.empty()) return false;
        Bytes bytes;
        if(!store_.load(key,bytes)) return false;
        Condition condition;
        if(!decode(bytes,condition) || condition.empty()) {store_.erase(key); return false;}
        output=std::move(condition); return true;
    }
    bool save(const std::string& key,const Condition& condition) const noexcept {
        try {
            if(key.empty()||condition.empty()||!store_.config().enabled()) return false;
            return store_.save(key,encode(condition,size_t(store_.config().entry_limit()-envelope_bytes)));
        } catch(...) {return false;}
    }
};
} // namespace wildimage_conditioning
#endif
