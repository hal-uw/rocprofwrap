#include <stdint.h>
#include "rocm_smi/rocm_smi.h"
#include <cstdio>

int main() {
  rsmi_status_t ret;
  uint32_t num_devices;
  uint16_t dev_id;

  ret = rsmi_init(0);
  ret = rsmi_num_monitor_devices(&num_devices);

  printf("Found %d devices\n",num_devices);

  for (int i=0; i < num_devices; ++i) {
    ret = rsmi_dev_id_get(i, &dev_id);
    // dev_id holds the device ID of device i, upon a
    // successful call
    printf("Device ID for device %d = %x\n",i,dev_id);
  }

  ret = rsmi_shut_down();
  return 0;
}
