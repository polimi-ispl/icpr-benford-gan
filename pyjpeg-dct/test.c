#include "jpeg.h"
#include <string.h>

void flip_in_place(JCOEFPTR ptr1) {
  for (int k = 0; k < DCTSIZE2; k+=2) {
    ptr1[k+1] = -ptr1[k+1];
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "no filename supplied\n");
    return 1;
  }
  FILE *infile;
  const char *filename = argv[1];
  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return 1;
  }
  struct jpeg_stuff stuff;
  readjpeg(infile, &stuff);

  int comps = stuff.cinfo.num_components;
  for (int comp = 0; comp < comps; comp++) {
    int maxy = stuff.cinfo.comp_info[comp].height_in_blocks;
    int maxx = stuff.cinfo.comp_info[comp].width_in_blocks;
    fprintf(stdout,"component %d width: %d height: %d\n", comp, maxx, maxy);
    for (int y = 0; y < maxy; y ++) {
      for (int x = 0; 2 * x < maxx; x++) {
        JCOEF tmp[DCTSIZE2];
        JCOEF tmp2[DCTSIZE2];
        JCOEFPTR ptr;
        ptr = get_coeff_block(&stuff, x, y, comp);
        flip_in_place(ptr);
        memcpy(tmp, ptr, BLOCKSIZE);
        ptr = get_coeff_block(&stuff, maxx - x - 1, y, comp);
        flip_in_place(ptr);
        memcpy(tmp2, ptr, BLOCKSIZE);
        memcpy(ptr, tmp, BLOCKSIZE);
        ptr = get_coeff_block(&stuff, x, y, comp);
        memcpy(ptr, tmp2, BLOCKSIZE);
      }
    }
  }
      
  fprintf(stdout, "done writing\n");

  FILE *outfile = fopen(argv[2], "wb");
  writejpeg(outfile, &stuff);
  fclose(outfile);
  closejpeg(&stuff);
  fclose(infile);
  return 0;
}
