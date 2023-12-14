extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .file("src/cuda/poker.cu")
        .compile("poker");
}
