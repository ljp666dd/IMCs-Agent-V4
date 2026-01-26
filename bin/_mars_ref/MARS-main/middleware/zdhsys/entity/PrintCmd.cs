using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.entity
{
    public class PrintCmd
    {
        public string device;
        public string cmd;
        /// <summary>
        /// 0 机械臂 1 反应设备
        /// </summary>
        public int opt;
        /// <summary>
        /// 反应设备参数
        /// </summary>
        public DevInfo devInfo;
    }
}
