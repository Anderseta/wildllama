#ifndef WILDIMAGE_CONDITIONING_STORE_H
#define WILDIMAGE_CONDITIONING_STORE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include "wildimage_ram_budget.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace wildimage_conditioning {
namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t,wildimage_ram::Allocator<uint8_t>>;
using Digest = std::array<uint8_t, 32>;
constexpr uint64_t entry_max_bytes = UINT64_C(256) << 20;
constexpr uint64_t store_max_bytes = UINT64_C(8192) << 20;
constexpr size_t envelope_bytes = 80;

class SHA256 {
    std::array<uint32_t, 8> state_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<uint8_t, 64> buffer_{};
    uint64_t bytes_ = 0;
    size_t used_ = 0;

    static uint32_t rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static constexpr uint32_t k[] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) | (uint32_t(p[4*i+2]) << 8) | p[4*i+3];
        }
        for (unsigned i = 16; i < 64; ++i) {
            const auto a = w[i-15], b = w[i-2];
            w[i] = w[i-16] + (rotate(a,7)^rotate(a,18)^(a>>3)) + w[i-7] + (rotate(b,17)^rotate(b,19)^(b>>10));
        }
        auto a=state_[0], b=state_[1], c=state_[2], d=state_[3], e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotate(e,6)^rotate(e,11)^rotate(e,25)) + ((e&f)^((~e)&g)) + k[i] + w[i];
            const uint32_t t2 = (rotate(a,2)^rotate(a,13)^rotate(a,22)) + ((a&b)^(a&c)^(b&c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

public:
    void update(const void* data, size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        bytes_ += size;
        while (size) {
            const size_t n = size < 64-used_ ? size : 64-used_;
            std::memcpy(buffer_.data()+used_, p, n);
            used_ += n; p += n; size -= n;
            if (used_ == 64) { block(buffer_.data()); used_ = 0; }
        }
    }
    Digest finish() const {
        SHA256 copy = *this;
        const uint64_t bits = bytes_ * 8;
        uint8_t pad[128]{}; pad[0] = 0x80;
        const size_t n = used_ < 56 ? 56-used_ : 120-used_;
        copy.update(pad, n);
        uint8_t tail[8];
        for (unsigned i=0; i<8; ++i) tail[7-i] = uint8_t(bits >> (8*i));
        copy.update(tail, 8);
        Digest digest{};
        for (unsigned i=0; i<8; ++i) for (unsigned j=0; j<4; ++j) digest[4*i+j] = uint8_t(copy.state_[i] >> (24-8*j));
        return digest;
    }
};

inline Digest sha256(const void* data, size_t size) { SHA256 h; h.update(data,size); return h.finish(); }
inline std::string hex(const Digest& digest) {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::string result(64,'0');
    for (size_t i=0; i<digest.size(); ++i) { result[2*i]=alphabet[digest[i]>>4]; result[2*i+1]=alphabet[digest[i]&15]; }
    return result;
}
inline bool valid_key(const std::string& key) {
    return key.size()==64 && std::all_of(key.begin(),key.end(),[](char c) { return (c>='0'&&c<='9')||(c>='a'&&c<='f'); });
}
inline void put64(uint8_t* p, uint64_t n) { for (unsigned i=0;i<8;++i) p[i]=uint8_t(n>>(8*i)); }
inline uint64_t get64(const uint8_t* p) { uint64_t n=0; for(unsigned i=0;i<8;++i) n|=uint64_t(p[i])<<(8*i); return n; }

struct Config {
    fs::path directory;
    std::string name_space;
    uint64_t budget = 0;
    bool enabled() const { return !directory.empty() && valid_key(name_space) && budget > envelope_bytes && budget <= store_max_bytes; }
    uint64_t entry_limit() const { return budget < entry_max_bytes ? budget : entry_max_bytes; }
    static Config environment() noexcept {
        try {
            Config config;
            const char* stamp = std::getenv("WILDIMAGE_CONDITION_NAMESPACE");
            const char* amount = std::getenv("WILDIMAGE_CONDITION_CACHE_BYTES");
            if (!stamp || !amount || !valid_key(stamp) || !*amount) return config;
            config.name_space = stamp;
            for (const char* p=amount; *p; ++p) {
                if (*p<'0'||*p>'9' || config.budget > (store_max_bytes-uint64_t(*p-'0'))/10) return Config{};
                config.budget=config.budget*10+uint64_t(*p-'0');
            }
#ifdef _WIN32
            const DWORD length=GetEnvironmentVariableW(L"WILDIMAGE_CONDITION_CACHE_DIR",nullptr,0);
            if (!length || length>32768) return Config{};
            std::wstring directory(length,L'\0');
            const DWORD copied=GetEnvironmentVariableW(L"WILDIMAGE_CONDITION_CACHE_DIR",directory.data(),length);
            if (!copied || copied>=length) return Config{};
            directory.resize(copied); config.directory=directory;
#else
            const char* directory=std::getenv("WILDIMAGE_CONDITION_CACHE_DIR");
            if (!directory || !*directory) return Config{};
            config.directory=fs::u8path(directory);
#endif
            if (!config.directory.is_absolute()) return Config{};
            return config;
        } catch (...) { return {}; }
    }
};

inline bool safe_regular(const fs::path& path) {
    std::error_code ec;
    if (fs::symlink_status(path,ec).type()!=fs::file_type::regular || ec) return false;
#ifdef _WIN32
    const DWORD attributes=GetFileAttributesW(path.c_str());
    if (attributes==INVALID_FILE_ATTRIBUTES || (attributes&FILE_ATTRIBUTE_REPARSE_POINT)) return false;
#endif
    return true;
}
inline bool safe_directory(const fs::path& path) {
    std::error_code ec;
    if (fs::symlink_status(path,ec).type()!=fs::file_type::directory || ec) return false;
#ifdef _WIN32
    const DWORD attributes=GetFileAttributesW(path.c_str());
    if (attributes==INVALID_FILE_ATTRIBUTES || (attributes&FILE_ATTRIBUTE_REPARSE_POINT)) return false;
#endif
    return true;
}

class File {
public:
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
#endif
    File()=default;
    File(const File&)=delete;
    File& operator=(const File&)=delete;
    ~File() { close(); }
    void close() {
#ifdef _WIN32
        if (handle!=INVALID_HANDLE_VALUE) { CloseHandle(handle); handle=INVALID_HANDLE_VALUE; }
#else
        if (handle>=0) { ::close(handle); handle=-1; }
#endif
    }
    bool open(const fs::path& path, int mode) {
#ifdef _WIN32
        handle=CreateFileW(path.c_str(),mode?GENERIC_READ|GENERIC_WRITE:GENERIC_READ,
            FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,mode==1?CREATE_NEW:mode==2?OPEN_ALWAYS:OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        if(handle==INVALID_HANDLE_VALUE) return false;
        BY_HANDLE_FILE_INFORMATION info{};
        if(!GetFileInformationByHandle(handle,&info) || (info.dwFileAttributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT)) || GetFileType(handle)!=FILE_TYPE_DISK) {close(); return false;}
#else
        handle=::open(path.c_str(),(mode?O_RDWR:O_RDONLY)|O_CLOEXEC|O_NOFOLLOW|(mode==1?O_CREAT|O_EXCL:mode==2?O_CREAT:0),0600);
        if(handle<0) return false;
        struct stat info{};
        if(fstat(handle,&info)!=0 || !S_ISREG(info.st_mode)) {close(); return false;}
#endif
        return true;
    }
    uint64_t size() const {
#ifdef _WIN32
        LARGE_INTEGER n{}; return GetFileSizeEx(handle,&n)&&n.QuadPart>=0?uint64_t(n.QuadPart):UINT64_MAX;
#else
        struct stat info{}; return fstat(handle,&info)==0&&info.st_size>=0?uint64_t(info.st_size):UINT64_MAX;
#endif
    }
    bool read(void* data,size_t size) {
        auto* p=static_cast<uint8_t*>(data);
        while(size) {
            const size_t n=size<(1u<<20)?size:(1u<<20);
#ifdef _WIN32
            DWORD done=0; if(!ReadFile(handle,p,DWORD(n),&done,nullptr)||!done) return false;
#else
            const auto done=::read(handle,p,n); if(done<=0) return false;
#endif
            p+=done; size-=size_t(done);
        }
        return true;
    }
    bool write(const void* data,size_t size) {
        const auto* p=static_cast<const uint8_t*>(data);
        while(size) {
            const size_t n=size<(1u<<20)?size:(1u<<20);
#ifdef _WIN32
            DWORD done=0; if(!WriteFile(handle,p,DWORD(n),&done,nullptr)||!done) return false;
#else
            const auto done=::write(handle,p,n); if(done<=0) return false;
#endif
            p+=done; size-=size_t(done);
        }
        return true;
    }
    bool flush() {
#ifdef _WIN32
        return FlushFileBuffers(handle)!=0;
#else
        return fsync(handle)==0;
#endif
    }
};

class Lock {
    File file_;
public:
    bool acquire(const fs::path& directory) {
        if(!file_.open(directory/"wildimage-conditioning.lock",2)) return false;
#ifdef _WIN32
        OVERLAPPED position{};
        return LockFileEx(file_.handle,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,1,0,&position)!=0;
#else
        return flock(file_.handle,LOCK_EX|LOCK_NB)==0;
#endif
    }
};

class Store {
    Config config_;
    struct Entry { fs::path path; uint64_t size; fs::file_time_type used; };
    static bool entry_name(const std::string& name) {
        return name.size()==73 && name.compare(0,5,"wic1-")==0 && name.compare(69,4,".bin")==0 && valid_key(name.substr(5,64));
    }
    static bool temporary_name(const std::string& name) {
        if(name.size()<76 || name.compare(0,5,"wic1-")!=0 || !valid_key(name.substr(5,64)) || name.compare(69,5,".tmp-")!=0) return false;
        return std::all_of(name.begin()+74,name.end(),[](char c){return (c>='0'&&c<='9')||c=='-';});
    }
    bool prepare(Lock& lock) const {
        if(!config_.enabled()) return false;
        std::error_code ec;
        fs::create_directories(config_.directory,ec);
        return !ec && safe_directory(config_.directory) && lock.acquire(config_.directory);
    }
    bool prune(uint64_t limit,const fs::path& keep={}) const {
        std::vector<Entry> entries;
        uint64_t total=0;
        std::error_code ec;
        fs::directory_iterator iterator(config_.directory,ec), end;
        if(ec) return false;
        for(;iterator!=end;iterator.increment(ec)) {
            if(ec) return false;
            const auto path=iterator->path();
            const auto name=path.filename().string();
            if((!entry_name(name)&&!temporary_name(name)) || !safe_regular(path)) continue;
            const auto size=fs::file_size(path,ec); if(ec) return false;
            if(temporary_name(name) || size>config_.entry_limit()) {
                if(!fs::remove(path,ec)||ec) return false;
                continue;
            }
            const auto used=fs::last_write_time(path,ec); if(ec || total>UINT64_MAX-size) return false;
            entries.push_back({path,size,used}); total+=size;
            if(entries.size()>4096) {
                auto oldest=std::min_element(entries.begin(),entries.end(),[&](const Entry& a,const Entry& b){
                    if(a.path==keep) return false;
                    if(b.path==keep) return true;
                    return a.used<b.used;
                });
                if(!fs::remove(oldest->path,ec)||ec) return false;
                total-=oldest->size; entries.erase(oldest);
            }
        }
        if(ec) return false;
        std::sort(entries.begin(),entries.end(),[&](const Entry& a,const Entry& b){
            if(a.path==keep) return false;
            if(b.path==keep) return true;
            return a.used==b.used?a.path<b.path:a.used<b.used;});
        for(const auto& entry:entries) {
            if(total<=limit) break;
            if(!fs::remove(entry.path,ec)||ec) return false;
            total-=entry.size;
        }
        return total<=limit;
    }

public:
    explicit Store(Config config=Config::environment()):config_(std::move(config)){}
    const Config& config() const {return config_;}
    fs::path path(const std::string& key) const {return config_.directory/("wic1-"+key+".bin");}
    void erase(const std::string& key) const noexcept {
        try {
            if(!valid_key(key)) return;
            Lock lock; if(!prepare(lock)) return;
            std::error_code ec;
            if(safe_regular(path(key))) fs::remove(path(key),ec);
        } catch(...) {}
    }
    bool load(const std::string& key,Bytes& payload) const noexcept {
        try {
            if(!valid_key(key)) return false;
            Lock lock; if(!prepare(lock)) return false;
            const auto source=path(key);
            if(!prune(config_.budget,source) || !safe_regular(source)) return false;
            File file; if(!file.open(source,0)) return false;
            const uint64_t size=file.size();
            bool valid=size>=envelope_bytes && size<=config_.entry_limit();
            std::array<uint8_t,envelope_bytes> header{};
            if(valid) valid=file.read(header.data(),header.size()) && std::memcmp(header.data(),"WICOND01",8)==0 && get64(header.data()+8)==size-envelope_bytes;
            const auto key_digest=sha256(key.data(),key.size());
            if(valid) valid=std::memcmp(header.data()+16,key_digest.data(),32)==0;
            Bytes bytes;
            if(valid) {bytes.resize(size_t(size-envelope_bytes)); valid=file.read(bytes.data(),bytes.size());}
            if(valid) {const auto checksum=sha256(bytes.data(),bytes.size()); valid=std::memcmp(header.data()+48,checksum.data(),32)==0;}
            file.close();
            std::error_code ec;
            if(!valid) {fs::remove(source,ec); return false;}
            fs::last_write_time(source,fs::file_time_type::clock::now(),ec);
            payload=std::move(bytes); return true;
        } catch(...) {return false;}
    }
    bool save(const std::string& key,const Bytes& payload) const noexcept {
        fs::path temporary;
        try {
            if(!valid_key(key)||!config_.enabled() || payload.size()>config_.entry_limit()-envelope_bytes) return false;
            Lock lock; if(!prepare(lock)) return false;
            std::error_code ec;
            const auto target=path(key);
            if(fs::exists(fs::symlink_status(target,ec))) return false;
            if(ec && ec!=std::errc::no_such_file_or_directory) return false;
            const uint64_t size=payload.size()+envelope_bytes;
            if(!prune(config_.budget-size)) return false;
            static std::atomic<uint64_t> serial{0};
#ifdef _WIN32
            const uint64_t process=GetCurrentProcessId();
#else
            const uint64_t process=uint64_t(getpid());
#endif
            temporary=config_.directory/("wic1-"+key+".tmp-"+std::to_string(process)+"-"+std::to_string(++serial));
            File file; if(!file.open(temporary,1)) return false;
            std::array<uint8_t,envelope_bytes> header{};
            std::memcpy(header.data(),"WICOND01",8); put64(header.data()+8,payload.size());
            const auto key_digest=sha256(key.data(),key.size()), checksum=sha256(payload.data(),payload.size());
            std::memcpy(header.data()+16,key_digest.data(),32); std::memcpy(header.data()+48,checksum.data(),32);
            const bool written=file.write(header.data(),header.size())&&file.write(payload.data(),payload.size())&&file.flush();
            file.close();
            bool renamed=false;
#ifdef _WIN32
            if(written) renamed=MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)!=0;
#else
            if(written) {fs::rename(temporary,target,ec); renamed=!ec;}
#endif
            if(!renamed) fs::remove(temporary,ec);
            return renamed;
        } catch(...) {std::error_code ec; if(!temporary.empty()) fs::remove(temporary,ec); return false;}
    }
};
} // namespace wildimage_conditioning
#endif
