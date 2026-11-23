// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../option_parser.h"

#ifndef ADDRDEC_H
#define ADDRDEC_H

#include "../abstract_hardware_model.h"

// Bit 49 of a PIM-flagged address selects the PIM half of the channels when
// -gpgpu_memory_partition_indexing is PIM (6).
#define PIM_ADDR_FLAG (1ULL << 49)

// Reserved DRAM rows of a PIM channel (single definition shared by the
// address decoder and the DRAM model). Activating PIM_ENTER_MRS_ROW /
// PIM_EXIT_MRS_ROW starts a NORMAL->PIM / PIM->NORMAL mode transition.
// PIM_GRF_ROW addresses the PIM general register file (e.g. input vector
// broadcast) and is not a PIM data row. The address mapping must provide
// enough row bits to represent all of them (checked at initialization).
#define PIM_ENTER_MRS_ROW 0x27FF
#define PIM_EXIT_MRS_ROW 0x2FFF
#define PIM_GRF_ROW 0x3FFF
#define PIM_MAX_RESERVED_ROW PIM_GRF_ROW

enum partition_index_function {
  CONSECUTIVE = 0,
  BITWISE_PERMUTATION,
  IPOLY,
  PAE,
  RANDOM,
  CUSTOM,
  PIM,
};

struct addrdec_t {
  void print(FILE *fp) const;

  unsigned chip;
  unsigned bk;
  unsigned row;
  unsigned col;
  unsigned burst;

  unsigned sub_partition;
};

class linear_to_raw_address_translation {
 public:
  linear_to_raw_address_translation();
  void addrdec_setoption(option_parser_t opp);
  void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel);

  // accessors
  void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const;
  new_addr_type partition_address(new_addr_type addr) const;

  // PIM channels exist only with the PIM partition indexing; they are the
  // upper half of the memory channels.
  bool pim_enabled() const { return memory_partition_indexing == PIM; }
  bool is_pim_channel(unsigned chip) const {
    return pim_enabled() && chip >= m_n_channel / 2;
  }
  // PIM command space: uncacheable, every access must reach the PIM channel.
  bool is_pim_address(new_addr_type addr) const {
    return pim_enabled() && (addr & PIM_ADDR_FLAG) != 0;
  }

 private:
  void addrdec_parseoption(const char *option);
  void sweep_test() const;  // sanity check to ensure no overlapping

  enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, N_ADDRDEC };

  const char *addrdec_option;
  int gpgpu_mem_address_mask;
  partition_index_function memory_partition_indexing;
  bool run_test;

  int ADDR_CHIP_S;
  unsigned char addrdec_mklow[N_ADDRDEC];
  unsigned char addrdec_mkhigh[N_ADDRDEC];
  new_addr_type addrdec_mask[N_ADDRDEC];
  new_addr_type sub_partition_id_mask;

  unsigned int gap;
  unsigned m_n_channel;
  int m_n_sub_partition_in_channel;
  int m_n_sub_partition_total;
  unsigned log2channel;
  unsigned log2sub_partition;
  unsigned nextPowerOf2_m_n_channel;
  unsigned m_row_bits;  // number of row bits in the address mapping
};

#endif
