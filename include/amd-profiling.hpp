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

int64_t currentVoltage;
int64_t tempEdgeG, tempJunction, tempMemory, tempHbm0, tempHbm1, tempHbm2,
    tempHbm3;

uint64_t power;
uint64_t prevEnergy = 0, currEnergy;
uint64_t startTime, timeStamp1, timeStamp2, etimeStamp, prevTimeStamp;
uint64_t profItr = 0;

float resolution;
float ePower = 0.0;

ThreadPool pool(1);

std::string hwCounters[8] = {
    "SQ_INSTS_VALU_MFMA_F16",      "SQ_INSTS_VALU_MFMA_F32",
    "SQ_INSTS_VALU_MFMA_F64",      "SQ_INSTS_VALU_MFMA_I8",
    "SQ_INSTS_VALU_MFMA_MOPS_F16", "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64", "SQ_INSTS_VALU_MFMA_MOPS_I8",
    // "SQ_LDS_IDX_ACTIVE","SQ_LDS_BANK_CONFLICT","TCP_TOTAL_CACHE_ACCESSES_sum",
    // "TCP_TCC_READ_REQ_sum","TCP_TCC_WRITE_REQ_sum","TCC_EA_RDREQ_DRAM_sum",
    // "TCC_EA_WRREQ_DRAM_sum","TCC_EA_RDREQ_32B_sum","TCC_EA_WRREQ_64B_sum"
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

  std::string header = "avg_power,curr_socket_power,power_from_e,vddgfx_volt,";
  // "gpu_busy_percent,mem_busy_percent,"
  // "vddgfx_volt,temp_edge,temp_junct,temp_mem,"
  // "temp_hbm0,temp_hbm1,temp_hbm2,temp_hbm3,"
  // "trg_sysclk,trg_dfclk,trg_dcefclk,trg_socclk,"
  // "trg_memclk,trg_pcieclk,"
  // "temp_vrgfx,temp_vrmem,temp_vrsoc,"
  // "throttle_status,curr_uclk,";

  output << header;

  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << "gfxclk" << i << ",";

  // }
  // for (int i = 0; i < RSMI_MAX_NUM_CLKS; i++){
  //     output << "socclk" << i << ",";

  // }
  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << "vclk" << i << ",";

  // }
  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << "dclk" << i << ",";

  // }
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

  if (profItr > 0) {
    ePower = resolution * (currEnergy - prevEnergy) / 1000000.0 /
             ((timeStamp1 - prevTimeStamp) / 1000000000.0);
  }
  // add the voltage here
  output << power / 1000000.0 << "," << currPower / 1000000.0 << "," << ePower << "," << currentVoltage << ",";
  // output << gpuBusyPercent << "," << memBusyPercent << "," << currentVoltage
  // << ","; output << tempEdge/1000.0 << "," << tempJunction/1000.0 << "," <<
  // tempMemory/1000.0 << "," << tempHbm0/1000.0 << "," << tempHbm1/1000.0 <<
  // ","; output << tempHbm2/1000.0 << "," << tempHbm3/1000.0 << ","; output <<
  // trgSysclk.frequency[trgSysclk.current]/1000000 << "," <<
  // trgDfclk.frequency[trgDfclk.current]/1000000 << ","; output <<
  // trgDcefclk.frequency[trgDcefclk.current]/1000000 << "," <<
  // trgSocclk.frequency[trgSocclk.current]/1000000 << ","; output <<
  // trgMemclk.frequency[trgMemclk.current]/1000000 << "," <<
  // trgPcieclk.frequency[trgPcieclk.current]/1000000 << ","; output <<
  // tempVrgfx << "," << tempVrmem << "," << tempVrsoc << ","; output<<
  // throttleStatus << "," << currUclk << ",";

  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << currGfxclk[i] << ",";
  // }
  // for (int i = 0; i < RSMI_MAX_NUM_CLKS; i++){
  //     output << currSocclk[i] << ",";
  // }
  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << currVclk[i] << ",";
  // }
  // for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; i++){
  //     output << currDclk[i] << ",";
  // }

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
  rsmi_dev_current_socket_power_get( device, &currPower );
  // rsmi_dev_metrics_energy_acc_get(device, &accEnergy);
  // rsmi_dev_busy_percent_get( device, &gpuBusyPercent );
  // rsmi_dev_memory_busy_percent_get( device, &memBusyPercent );
  rsmi_dev_volt_metric_get(device, RSMI_VOLT_TYPE_VDDGFX, RSMI_VOLT_CURRENT,&currentVoltage); 
  //rsmi_dev_temp_metric_get( device, RSMI_TEMP_TYPE_EDGE,
  // RSMI_TEMP_CURRENT, &tempEdge ); rsmi_dev_temp_metric_get( device,
  // RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &tempJunction );
  // rsmi_dev_temp_metric_get( device, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT,
  // &tempMemory ); rsmi_dev_temp_metric_get( device, RSMI_TEMP_TYPE_HBM_0,
  // RSMI_TEMP_CURRENT, &tempHbm0 ); rsmi_dev_temp_metric_get( device,
  // RSMI_TEMP_TYPE_HBM_1, RSMI_TEMP_CURRENT, &tempHbm1 );
  // rsmi_dev_temp_metric_get( device, RSMI_TEMP_TYPE_HBM_2, RSMI_TEMP_CURRENT,
  // &tempHbm2 ); rsmi_dev_temp_metric_get( device, RSMI_TEMP_TYPE_HBM_3,
  // RSMI_TEMP_CURRENT, &tempHbm3 ); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_SYS, &trgSysclk); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_DF, &trgDfclk); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_DCEF, &trgDcefclk); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_SOC, &trgSocclk); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_MEM, &trgMemclk); rsmi_dev_gpu_clk_freq_get( device,
  // RSMI_CLK_TYPE_PCIE, &trgPcieclk); rsmi_dev_metrics_temp_vrgfx_get( device,
  // &tempVrgfx ); rsmi_dev_metrics_temp_vrmem_get( device, &tempVrmem );
  // rsmi_dev_metrics_temp_vrsoc_get( device, &tempVrsoc );
  // rsmi_dev_metrics_throttle_status_get( device,  &throttleStatus );
  // rsmi_dev_metrics_curr_uclk_get( device, &currUclk );
  // rsmi_dev_metrics_curr_gfxclk_get( device, &currGfxclk );
  // rsmi_dev_metrics_curr_socclk_get( device, &currSocclk );
  // rsmi_dev_metrics_curr_vclk0_get( device, &currVclk );
  // rsmi_dev_metrics_curr_dclk0_get( device, &currDclk );

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
