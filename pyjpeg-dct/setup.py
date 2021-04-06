from distutils.core import setup, Extension

module1 = Extension('jpeg',
    sources = ['jpegmodule.c', 'jpeg.c'],
    libraries = ['jpeg'],
    extra_compile_args = ['-Wextra', '-Wno-unused-parameter', '-Wno-missing-field-initializers']
    )

setup (name = 'JpegPackage',
    version = '0.1',
    description = 'jpeg mangler',
    ext_modules = [module1])
