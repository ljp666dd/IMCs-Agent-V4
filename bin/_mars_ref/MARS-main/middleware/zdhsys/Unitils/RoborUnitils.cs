using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Unitils
{
    public class RoborUnitils
    {
        public static TCPHelper robot;
        public static bool ConnectRobot(string ip,int port)
        {
            robot = new TCPHelper();
            robot.Connect(ip, port);
            if (robot.Connected())
            {
                return true;
            }
            return false;
        }        

    }
}
