#include "rocm_smi/rocm_smi.h"
#include "rocprofiler/v2/rocprofiler.h"
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <thread_pool.h>
#include <unistd.h>

#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>

// Global variables
std::ofstream output;
rsmi_status_t ret;
const char* status_string;
hipDeviceProp_t devProp;
rocprofiler_session_id_t dp_session_id;

std::vector<const char *> counters;

std::vector<rocprofiler_device_profile_metric_t> profiling_data;

int duration, stop = 0;

uint32_t device;
uint32_t gpuBusyPercent, memBusyPercent;
uint32_t throttleStatus;

uint16_t currUclk;
uint16_t tempVrgfx, tempVrmem, tempVrsoc;
uint64_t currPower;
uint64_t accEnergy;

rsmi_frequencies_t trgSysclk, trgDfclk, trgDcefclk, trgSocclk, trgMemclk,
    trgPcieclk;

int64_t tempEdgeG, tempJunction, tempMemory;

uint64_t power;
uint64_t prevEnergy = 0, currEnergy;
uint64_t startTime, timeStamp1, timeStamp2, etimeStamp, prevTimeStamp;
uint64_t profItr = 0;

float resolution;
float ePower = 0.0;

ThreadPool pool(1);

std::string hwCounters[2] = {
    "SQ_CYCLES", "SQ_BUSY_CYCLES",
};

// Functions

template <class CharContainer>
size_t file_get_contents(const char *filename, CharContainer *v) {

  ::FILE *fp = ::fopen(filename, "rb");
  C4_CHECK_MSG(fp != nullptr, "could not open file");
  ::fseek(fp, 0, SEEK_END);
  long sz = ::ftell(fp);
  v->resize(static_cast<typename CharContainer::size_type>(sz));
  if (sz) {
    ::rewind(fp);
    size_t ret = ::fread(&(*v)[0], 1, v->size(), fp);
    C4_CHECK(ret == (size_t)sz);
  }
  ::fclose(fp);
  return v->size();
}

void parseYaml(const char *filename) {

  std::string contents;
  file_get_contents<std::string>(filename, &contents);
  ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(contents));
  std::cout << tree[0]["gpu-ids"].num_children() << std::endl;
}

void hwCounterInit() {

  // Only 8 counters can be collected from the SQ block.
  /*
  counters.emplace_back("SQ_INSTS_VALU_ADD_F16");
  counters.emplace_back("SQ_INSTS_VALU_MUL_F16");
  counters.emplace_back("SQ_INSTS_VALU_FMA_F16");
  counters.emplace_back("SQ_INSTS_VALU_TRANS_F16");
  */

  /*
  counters.emplace_back("SQ_INSTS_VALU_ADD_F32");
  counters.emplace_back("SQ_INSTS_VALU_MUL_F32");
  counters.emplace_back("SQ_INSTS_VALU_FMA_F32");
  /*
  counters.emplace_back("SQ_INSTS_VALU_TRANS_F32");

  counters.emplace_back("SQ_INSTS_VALU_ADD_F64");
  counters.emplace_back("SQ_INSTS_VALU_MUL_F64");
  counters.emplace_back("SQ_INSTS_VALU_FMA_F64");
  counters.emplace_back("SQ_INSTS_VALU_TRANS_F64");
  */
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_F16");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_F32");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_F64");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_I8");

  // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_F16");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_F32");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_F64");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_I8");
  // just emplace from the hw_counters set as the same order
  // traverse the hw_counters and emplace them into the counters
  for (auto &counter : hwCounters) {
    counters.emplace_back(counter.c_str());
  }

  // FLOPS
  // counters.emplace_back("SQ_INSTS_VALU_ADD_F32");
  // counters.emplace_back("SQ_INSTS_VALU_MUL_F32");
  // counters.emplace_back("SQ_INSTS_VALU_FMA_F32");
  // // counters.emplace_back("SQ_INSTS_VALU_TRANS_F32");
  // // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_BF16");
  // counters.emplace_back("SQ_INSTS_VALU_MFMA_MOPS_F32");

  // // LDS
  // counters.emplace_back("SQ_LDS_IDX_ACTIVE");
  // counters.emplace_back("SQ_LDS_BANK_CONFLICT");

  // // L1
  // counters.emplace_back("TCP_TOTAL_CACHE_ACCESSES_sum");

  // // L2
  // counters.emplace_back("TCP_TCC_READ_REQ_sum");
  // counters.emplace_back("TCP_TCC_WRITE_REQ_sum");

  // // HBM
  // counters.emplace_back("TCC_EA_RDREQ_DRAM_sum");
  // counters.emplace_back("TCC_EA_WRREQ_DRAM_sum");
  // counters.emplace_back("TCC_EA_RDREQ_32B_sum");
  // counters.emplace_back("TCC_EA_WRREQ_64B_sum");

  profiling_data.resize(counters.size());
}

void signal_callback_handler(int signum) { stop = 1; }

void header(std::ofstream &output) {

  std::string header = "avg_power,curr_socket_power,energy_acc,"
    "temp_edge,temp_junct,temp_mem,"
    "gpu_busy_percent,mem_busy_percent,"
    "trg_sysclk,trg_dfclk,trg_dcefclk,trg_socclk,"
    "trg_memclk,trg_pcieclk,";

  output << header;

  // for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //     output << "xgmi_read_link" << i << ",";

  // }
  // for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //     output << "xgmi_write_link" << i << ",";

  // }

  for (int i = 0; i < counters.size(); i++) {
    output << hwCounters[i] << ",";
  }

  output << "timestamp\n";
}

void writeData(std::ofstream &output) {

  output << power / 1000000.0 << "," << currPower / 1000000.0 << "," << currEnergy << ",";
  output << tempEdgeG/1000.0 << "," << tempJunction/1000.0 << "," << tempMemory/1000.0 << ",";
  output << gpuBusyPercent << "," << memBusyPercent << ",";
  output << trgSysclk.frequency[trgSysclk.current]/1000000 << "," << trgDfclk.frequency[trgDfclk.current]/1000000 << ",";
  output << trgDcefclk.frequency[trgDcefclk.current]/1000000 << "," << trgSocclk.frequency[trgSocclk.current]/1000000 << ",";
  output << trgMemclk.frequency[trgMemclk.current]/1000000 << "," << trgPcieclk.frequency[trgPcieclk.current]/1000000 << ",";

  // if (profItr > 0 ){
  //     for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //         output << currXgmiRead[i] - prevXgmiRead[i] << ",";
  //     }
  //     for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //         output << currXgmiWrite[i] - prevXgmiWrite[i] << ",";
  //     }
  // }
  // else{
  //     for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //         output << 0 << ",";
  //     }
  //     for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; i++){
  //         output << 0 << ",";
  //     }
  // }

  for (int i = 0; i < profiling_data.size(); i++) {
    output << (uint64_t)profiling_data[i].value.value << ",";
  }

  output << timeStamp1 << "\n";
  profItr += 1;
}

void getData() {

  // rocprof sampling performance hw counters
  pool.enqueue(rocprofiler_device_profiling_session_poll, dp_session_id,
               &profiling_data[0]);

  // rsmi sampling gpu metrics
  rsmi_dev_power_ave_get(device, 0, &power);
  rsmi_dev_energy_count_get(device, &currEnergy, &resolution, &etimeStamp);
  rsmi_dev_current_socket_power_get(device, &currPower);
  rsmi_dev_temp_metric_get(device, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &tempEdgeG);
  rsmi_dev_temp_metric_get(device, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &tempJunction);
  rsmi_dev_temp_metric_get(device, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT, &tempMemory);
  rsmi_dev_busy_percent_get( device, &gpuBusyPercent );
  rsmi_dev_memory_busy_percent_get( device, &memBusyPercent );
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_SYS, &trgSysclk);
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_DF, &trgDfclk);
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_DCEF, &trgDcefclk);
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_SOC, &trgSocclk);
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_MEM, &trgMemclk);
  rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_PCIE, &trgPcieclk);

  pool.wait();
}

void updateHist() {

  prevEnergy = currEnergy;
  prevTimeStamp = timeStamp1;
  // std::copy(currXgmiRead,
  // currXgmiRead+sizeof(currXgmiRead)/sizeof(currXgmiRead[0]), prevXgmiRead);
  // std::copy(currXgmiWrite,
  // currXgmiWrite+sizeof(currXgmiWrite)/sizeof(currXgmiWrite[0]),
  // prevXgmiWrite);
}