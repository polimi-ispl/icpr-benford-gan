#include "jpeg.h"
#include <stdio.h>

char writejpeg(FILE *outfile, struct jpeg_stuff *src) {
  struct jpeg_decompress_struct *injpeg = &src->cinfo;
  jvirt_barray_ptr *src_coeffs = src->src_coeffs;
  
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  jpeg_copy_critical_parameters(injpeg, &cinfo);
  if (jerr.num_warnings != 0) {
    return -1;
  }
  cinfo.optimize_coding = TRUE;
  jpeg_stdio_dest(&cinfo, outfile);
  if (jerr.num_warnings != 0) {
    return -1;
  }
  jpeg_write_coefficients(&cinfo, src_coeffs);
  if (jerr.num_warnings != 0) {
    return -1;
  }
  jpeg_finish_compress(&cinfo);
  if (jerr.num_warnings != 0) {
    return -1;
  }
  return 0;
}

char readjpeg(FILE *infile, struct jpeg_stuff *dest) {
  struct jpeg_decompress_struct *cinfo = &dest->cinfo;
  cinfo->err = jpeg_std_error(&dest->jerr);
  jpeg_create_decompress(cinfo);
  jpeg_stdio_src(cinfo, infile);
  if (cinfo->err->num_warnings != 0) {
    return -1;
  }
  char magic[2];
  size_t len = fread(&magic, 1, 2, infile);
  if (len != (size_t) 2 || magic[0] != (char)0xff || magic[1] != (char)0xd8) {
    return -1;
  }
  fseek(infile, 0, SEEK_SET);
  int header_status = jpeg_read_header(cinfo, FALSE);
  if (header_status != JPEG_HEADER_OK || cinfo->err->num_warnings != 0) {
    return -1;
  }
  jpeg_core_output_dimensions(cinfo);
  if (cinfo->err->num_warnings != 0) {
    return -1;
  }

  jvirt_barray_ptr *coeffs = jpeg_read_coefficients(cinfo);
  if (cinfo->err->num_warnings != 0) {
    return -1;
  }
  dest->src_coeffs = coeffs;

  return 0;

}
void closejpeg(struct jpeg_stuff *src) {
  jpeg_finish_decompress(&src->cinfo);
  jpeg_destroy_decompress(&src->cinfo);
  // should dispose of associated block arrays
}

char check_block_coords(struct jpeg_stuff *stuff, int x, int y, int comp) {
  struct jpeg_decompress_struct *cinfo = &stuff->cinfo;
  if (comp < 0 || comp > cinfo->num_components) {
    return -1;
  }
  jpeg_component_info *ji = cinfo->comp_info + comp;
  if (y < 0 || ((unsigned int) y) >= ji->height_in_blocks) {
    return -1;
  }
  if (x < 0 || ((unsigned int) x) >= ji->width_in_blocks) {
    return -1;
  }
  return 0;
}

JBLOCKARRAY get_coeff_barray_rows_internal(struct jpeg_stuff *stuff, int y, int comp, unsigned int rowcount) {
  struct jpeg_decompress_struct *cinfo = &stuff->cinfo;
  return cinfo->mem->access_virt_barray((j_common_ptr)cinfo,
      stuff->src_coeffs[comp], y, rowcount, TRUE);
}

JBLOCKARRAY get_coeff_barray_rows(struct jpeg_stuff *stuff, int y, int comp, unsigned int rowcount) {
  if (check_block_coords(stuff, 0, y, comp) != 0) {
    return NULL;
  }
  if (y + rowcount > stuff->cinfo.comp_info[comp].height_in_blocks) {
    return NULL;
  }
  return get_coeff_barray_rows_internal(stuff, y, comp, rowcount);
}

JCOEFPTR get_coeff_block(struct jpeg_stuff *stuff, int x, int y, int comp) {
  if (check_block_coords(stuff, x, y, comp) != 0) {
    return NULL;
  }
  jpeg_component_info *ji = stuff->cinfo.comp_info + comp;
  int rowindex = y % ji->v_samp_factor;
  int row = y - rowindex;
  JBLOCKARRAY jb = get_coeff_barray_rows_internal(stuff, row, comp, ji->v_samp_factor);
  if (jb == NULL) {
    return NULL;
  }
  return jb[rowindex][x];
}
