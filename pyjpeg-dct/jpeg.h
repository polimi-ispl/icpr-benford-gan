#include <stdio.h>
#include <jpeglib.h>
#include <jmorecfg.h>

#define BLOCKSIZE (sizeof(JCOEF) / sizeof(char)) * DCTSIZE2

struct jpeg_stuff {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  jvirt_barray_ptr *src_coeffs;
};

char writejpeg(FILE *outfile, struct jpeg_stuff *stuff);
char readjpeg(FILE *infile, struct jpeg_stuff *stuff);
void closejpeg(struct jpeg_stuff *src);
JBLOCKARRAY get_coeff_barray_rows(struct jpeg_stuff *stuff, int y, int comp, unsigned int rowcount);
JCOEFPTR get_coeff_block(struct jpeg_stuff *stuff, int x, int y, int comp);
