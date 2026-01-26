using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using zdhsys.Bean;

namespace zdhsys.entity
{
    public class DevInfo
    {
        /// <summary>
        /// 设备ID
        /// </summary>
        public int Id;
        /// <summary>
        /// 流程ID
        /// </summary>
        public int Index;
        public List<DeviceFieldsModel> Dfms;
    }
}
