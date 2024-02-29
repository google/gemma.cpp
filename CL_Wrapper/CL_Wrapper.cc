#include "CL_Wrapper.h"


std::vector<std::pair<std::string, std::string>>
cl_acc::get_platform_device_list() {
  std::vector<std::pair<std::string, std::string>> pla_dev;
  std::vector<Platform> pvec;
  Platform::get(&pvec);
  for (int i = 0; i < pvec.size(); ++i) {
    std::string pname;
    std::vector<Device> dvec;
    pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
    pvec[i].getDevices(CL_DEVICE_TYPE_ALL, &dvec);
    for (int j = 0; j < dvec.size(); ++j) {
      std::pair<std::string, std::string> dpair;
      std::string dname;
      dvec[j].getInfo(CL_DEVICE_NAME, &dname);
      dpair.first = pname;
      dpair.second = dname;
      pla_dev.push_back(dpair);
    }
  }
  return pla_dev;
}
void cl_acc::set_platform_device(const std::string& platform, 
                                     const std::string& device) {
  std::vector<Platform> pvec;
  Platform::get(&pvec);
  for (int i = 0; i < pvec.size(); ++i) {
    std::string pname;
    pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
    if (pname == platform) {
      this->PF = pvec[i];
      break;
    }
  }
  std::vector<Device> dvec;
  this->PF.getDevices(CL_DEVICE_TYPE_ALL, &dvec);
  for (int i = 0; i < dvec.size(); ++i) {
    std::string dname;
    dvec[i].getInfo(CL_DEVICE_NAME, &dname);
    if (dname == device) {
      this->DV = dvec[i];
      break;
    }
  }
  this->CT = Context(this->DV);
}

cl_acc::cl_acc(const std::string& platform, const std::string& device) {
  set_platform_device(platform, device);
}

