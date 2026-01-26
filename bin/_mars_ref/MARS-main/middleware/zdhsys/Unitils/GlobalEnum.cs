using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Unitils
{
    public class GlobalEnum
    {
        public enum UnitOfMeasurement
        {
            克,
            毫升,
            摄氏度,
            秒,
            分,
            次
        }

        public enum UnitBottle
        {
            大瓶子,
            小瓶子
        }
        /// <summary>
        /// 设备类别--指定通信协议
        /// </summary>
        public enum UnitDeviceType
        {
            加粉液设备,
            清洗设备,
            称量设备,
            包硅反应设备,
            机械臂,
            工作站,
            核酸检测仪
        }

    }
}
