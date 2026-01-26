using EasyModbus;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace zdhsys
{
   internal partial class Dev_BG
   {
        private Process pythonprocess;

        public void stopPythonDet()
        {
            // 当您想要终止 Python 脚本时
            if (!pythonprocess.HasExited)
            {
                pythonprocess.Kill(); // 终止 Python 进程
            }

            Console.WriteLine("Python 进程已终止");
        }

        public void startPythonDet()
        {
            // Python 解释器的路径
            string pythonInterpreter = @"D:\python311\python.exe"; // 替换为您的 Python 解释器路径

            // Python 脚本的路径
            string scriptPath = @"D:\botdet\databot\bot.py";

            // 创建并启动 Python 进程
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = pythonInterpreter,
                Arguments = scriptPath,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            pythonprocess = new Process { StartInfo = start };
            pythonprocess.Start();

            // 在这里添加代码，根据您的需求执行其他任务
            // 例如，等待一段时间
            //    Thread.Sleep(10000); // 等待10秒

            // 当您想要终止 Python 脚本时
            //    if (!pythonprocess.HasExited)
            //     {
            //          pythonprocess.Kill(); // 终止 Python 进程
            //      }

            //   Console.WriteLine("Python 进程已终止");
        }


        public void getCmdfromUDP()
        {
            // UDP 端口
            int UDP_PORT = 5006;
            // 创建一个 UdpClient 用于读取传入的数据
            UdpClient udpClient = new UdpClient(UDP_PORT);
            try
            {
                while (true)
                {
                    // 获取发送方的 IP 和端口
                    IPEndPoint RemoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);

                    // 接收来自发送方的数据
                    Byte[] receiveBytes = udpClient.Receive(ref RemoteIpEndPoint);
                    string returnData = Encoding.ASCII.GetString(receiveBytes);
                    // 检查接收到的数据是否为空
                    if (!string.IsNullOrWhiteSpace(returnData))
                    {
                        var receivedData = JsonConvert.DeserializeObject<Dictionary<int, List<object>>>(returnData);
                        if (receivedData != null && receivedData.Count > 0)
                        {
                            // 处理接收到的数据
                            //  Console.WriteLine("Received: " + returnData);
                     //       DealWithCmd(receivedData);
                        }
                        else
                        {
                            // 处理空数据
                            // Console.WriteLine("Received empty data");
                        }
                    }
                    else
                    {
                        Console.WriteLine("Received an empty message");
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }

        public void DealWithCmd(string receivedData) { 

        }



        public class bg_in_out_water
        {
            IPEndPoint Vaves_Ctl_IP;//电磁阀的IP
            int Vaves_Ctl_IP_port = 8234;//电磁阀通信端口
            ModbusClient m_modbusClient;

            public struct pod_ctl_st
            {
                public int podInPosSta;//瓶子在位状态
                public int waterInVaveSTA;//进阀门状态
                public int waterOutVaveSTA;//出阀门状态
                public float waterLeve;//液面高度百分比
            }


            public pod_ctl_st[] m_PodSTA = new pod_ctl_st[12];
            Thread th_readPod;

            private void UpdatePodStatus(Dictionary<int, List<object>> bottleData)
            {
                foreach (var item in bottleData)
                {
                    int bottleNumber = item.Key - 1; // 减1是因为数组索引从0开始
                    var bottleInfo = item.Value;

                    // 更新瓶子状态
                    m_PodSTA[bottleNumber].podInPosSta = Convert.ToInt32(bottleInfo[0]);
                    m_PodSTA[bottleNumber].waterLeve = Convert.ToSingle(bottleInfo[1]);

                    //     Console.WriteLine($"Bottle {bottleNumber + 1}: InPosSta = {m_PodSTA[bottleNumber].podInPosSta}, WaterLevel = {m_PodSTA[bottleNumber].waterLeve}");
                }
            }

            public void getpodstafromUDP()
            {
                // UDP 端口
                int UDP_PORT = 5005;
                // 创建一个 UdpClient 用于读取传入的数据
                UdpClient udpClient = new UdpClient(UDP_PORT);
                try
                {
                    while (true)
                    {
                        // 获取发送方的 IP 和端口
                        IPEndPoint RemoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);

                        // 接收来自发送方的数据
                        Byte[] receiveBytes = udpClient.Receive(ref RemoteIpEndPoint);
                        string returnData = Encoding.ASCII.GetString(receiveBytes);
                        // 检查接收到的数据是否为空
                        if (!string.IsNullOrWhiteSpace(returnData))
                        {
                            var receivedData = JsonConvert.DeserializeObject<Dictionary<int, List<object>>>(returnData);
                            if (receivedData != null && receivedData.Count > 0)
                            {
                                // 处理接收到的数据
                                //  Console.WriteLine("Received: " + returnData);
                                UpdatePodStatus(receivedData);
                            }
                            else
                            {
                                // 处理空数据
                                // Console.WriteLine("Received empty data");
                            }
                        }
                        else
                        {
                            Console.WriteLine("Received an empty message");
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                }
            }


            public bg_in_out_water() //初始化函数
            {
                // 创建Modbus客户端实例
                m_modbusClient = new ModbusClient("192.168.1.7", 8234); // 替换为你的设备IP地址和端口号
                m_modbusClient.Connect();
                Console.WriteLine("Connected to device");
                GetVaves_ALL_status();
                // 写入单个线圈（例如，地址1，状态true）
                Thread.Sleep(1000);
                for (int i = 1; i <= 24; i++)
                {
                    SetVaves_status(i, 0);
                }

                Console.WriteLine("Coil at address 1 written.");

                th_readPod = new Thread(getpodstafromUDP);//创建读取线程
                th_readPod.Start();

            }
            public bool GetVaves_ALL_status()
            {
                // 读取保持寄存器（例如，从地址0开始读取10个寄存器）
                int[] registers = m_modbusClient.ReadHoldingRegisters(0, 24); // 替换为你的寄存器地址和数量
                if (registers != null)
                {
                    // Console.WriteLine("Read " + registers.Length + " registers:");
                    int ID = 1;
                    foreach (int register in registers)
                    {
                        //     Console.WriteLine("ID=" + ID + "  data=" + register);

                        //  m_PodSTA[ID-1].waterOutVaveSTA = false;//更新阀门位置的函数
                        //    m_PodSTA[ID-1].waterInVaveSTA = false;
                        ID++;
                    }
                    for (int i = 0; i < 12; i++)
                    {
                        ID = i + 1;
                        m_PodSTA[i].waterInVaveSTA = registers[i * 2];
                        m_PodSTA[i].waterOutVaveSTA = registers[i * 2 + 1];
                    }
                }
                else
                {
                    Console.WriteLine("Failed to read registers.");
                }

                return true;
            }
            public bool SetVaves_status(int ID, int val)
            {
                if (ID <= 0 || ID > 24)
                {
                    return false;
                }
                if (val != 0 && val != 1)
                {
                    return false;
                }
                try
                {
                    m_modbusClient.WriteSingleRegister(ID - 1, val);
                    Console.WriteLine($"Coil at address {ID}={val}  written.");
                }
                catch (Exception e)
                {
                    return false;
                }
                GetVaves_ALL_status();
                return true;
            }

            public bool outWater(int ID, float dstvalue)//放水直到dst停止，ID从1开始-12
            {
                return true;
            }

            public bool inWater(int ID, float dstvalue)//加水直到dstvalue停止，ID从1开始-12
            {
                return true;
            }
        }

       


        public void pod1InWater(int pmpspd)
        {
            //阀门控制
            sub_dev_Vave.SetVaves_status(1, 1);
            //通道1开启
            start_pump_in(1, 0);
            Thread.Sleep(1000);
            start_pump(1, pmpspd);//泵控制
        }
        public void pod1InWaterStop()
        {
            stopPump(1);//泵控制
            Thread.Sleep(1000);
            //阀门控制
            sub_dev_Vave.SetVaves_status(1, 0);
        }
        public void pod1OutWater(int pmpspd)
        {
            //阀门控制
            sub_dev_Vave.SetVaves_status(2, 1);
            //通道1开启
            start_pump_out(1, 0);
            Thread.Sleep(1000);
            start_pump(1, pmpspd);//泵控制
        }
        public void pod1OutWaterStop()
        {
            stopPump(1);//泵控制
            Thread.Sleep(1000);
            //阀门控制
            sub_dev_Vave.SetVaves_status(2, 0);
        }

        public float targetlvin = 50;
        public float targetlvout = 30;
        private void processinwater()
        {
            float currentlv = sub_dev_Vave.m_PodSTA[0].waterLeve;
            if (currentlv - targetlvin > 0)
            {
                Console.WriteLine("target reached! return!");
                return;
            }
            Console.WriteLine("start in water!");
            pod1InWater(500);
            // this.pBVpod1pump.BackColor = Color.LightGreen;
            while (currentlv - targetlvin < -2)
            {
                currentlv = sub_dev_Vave.m_PodSTA[0].waterLeve;
                Thread.Sleep(100);//每隔100 检查一次
            }
            Console.WriteLine("in target reached, stop!");
            pod1InWaterStop();
            pod1OutWaterStop();
            //this.pBVpod1pump.BackColor = Color.DarkGray;

        }
        Thread p1inth;
        public void runpod1InWater()
        {
            if (p1inth != null)
            {
                if (p1inth.IsAlive)
                {
                    p1inth.Abort();
                    p1inth = null;
                }
            }
            p1inth = new Thread(processinwater);
            p1inth.Start();
        }

        public void runpod1OutWater()
        {
            if (p1inth != null)
            {
                if (p1inth.IsAlive)
                {
                    p1inth.Abort();
                    p1inth = null;
                }
            }
            p1inth = new Thread(processoutwater);
            p1inth.Start();

        }


        private void processoutwater()
        {
            float currentlv = sub_dev_Vave.m_PodSTA[0].waterLeve;
            if (currentlv - targetlvout < 0)
            {
                Console.WriteLine("target reached! return!");
                return;
            }
            Console.WriteLine("start out water!");
            pod1OutWater(500);
            //  this.pBVpod1pump.BackColor = Color.LightGreen;
            while (currentlv - targetlvout > 2)
            {
                currentlv = sub_dev_Vave.m_PodSTA[0].waterLeve;
                Thread.Sleep(100);//每隔100 检查一次
            }
            Console.WriteLine("out target reached, stop!");
            pod1InWaterStop();
            pod1OutWaterStop();
            //this.pBVpod1pump.BackColor = Color.DarkGray;
        }
        public int duliQuitFlag = 0;
        public void runInWaterAuto_duli(int ID, int pumpspeed,float targetLv)
        {
            if(ID<=0 || ID > 12) { return; }
            duliQuitFlag = 0;
            int i = ID - 1;
            float currentlv = sub_dev_Vave.m_PodSTA[i].waterLeve;
            float start_lv = currentlv;

            Stopwatch stopwatch = new Stopwatch();
            if (currentlv - targetLv > 0)
            {
                Console.WriteLine("ID="+ID.ToString()+" target reached"+ " "+targetLv.ToString() +" return!");
                return;
            }
         
            //打开进水阀门
            int openvavID = 2 * i + 1;
            int zuPumpID = (int)(i / 4) + 1;
            Console.WriteLine("进水 ID=" + ID.ToString() + " 阀门=" + openvavID.ToString() + " 泵+阀=" + zuPumpID.ToString());

            sub_dev_Vave.SetVaves_status(openvavID, 1);
            //切换进水组电磁阀
            start_pump_in(zuPumpID, 0);
            Thread.Sleep(500);
            //启动进水泵
            stopwatch.Start();
            start_pump(zuPumpID, pumpspeed);//泵控制

            while (currentlv - targetLv < -2)
            {
                currentlv = sub_dev_Vave.m_PodSTA[i].waterLeve;
                Thread.Sleep(100);//每隔100 检查一次
                if(duliQuitFlag==1)
                {
                    Console.WriteLine("inwater forced quit!");
                    return;
                }
            }
            stopwatch.Stop();
            TimeSpan elapsedTime = stopwatch.Elapsed;

            Console.WriteLine($"ID={ID}, from lv={start_lv} to lv={targetLv} with pumpspeed{pumpspeed}, " +
                $"the time is{elapsedTime},out target reached, stop! ");
            //关闭进水泵
            stopPump(zuPumpID);//泵控制

            Thread.Sleep(1000);
            //阀门控制
            sub_dev_Vave.SetVaves_status(openvavID, 0);
            //关闭阀门
            Console.WriteLine();

        }

        public int in159Flag = 0;
        //1 5 9 inwater
        public void runInWater159(int pumpspeed, float targetlv)
        {
            in159Flag = 0;
            float curlv1 = sub_dev_Vave.m_PodSTA[0].waterLeve;
            float curlv5 = sub_dev_Vave.m_PodSTA[4].waterLeve;
            float curlv9 = sub_dev_Vave.m_PodSTA[8].waterLeve;
            int[] addwaterSTA = new int[3];
            //int[] pumpSTA = new int[3];

            addwaterSTA[0] = 0;
            addwaterSTA[1] = 0;
            addwaterSTA[2] = 0;

            if(curlv1 - targetlv >0)
            {
                Console.WriteLine("ID= 1 target reached" + " " + curlv1.ToString() + " return!");
                addwaterSTA[0] = 1;
            }
            if (curlv5 - targetlv > 0)
            {
                Console.WriteLine("ID= 5 target reached" + " " + curlv5.ToString() + " return!");
                addwaterSTA[1] = 1;
            }
            if (curlv9 - targetlv > 0)
            {
                Console.WriteLine("ID= 9 target reached" + " " + curlv9.ToString() + " return!");
                addwaterSTA[2] = 1;
            }
            for(int i = 0;i<3;i++)
            {
                if (addwaterSTA[i] == 0)
                {
                    int openvavID_in = 8 * i + 1;
                    sub_dev_Vave.SetVaves_status(openvavID_in, 1);//设置进出水电磁阀
                    Thread.Sleep(100);
                    int zuPumpID = i +1;
                    start_pump_in(zuPumpID, 0);//设置泵为进水
                    Thread.Sleep(500);
                    //
                }
            }
            //开起泵
            for (int i = 0; i < 3; i++)
            {
                if (addwaterSTA[i] == 0)
                {
                    int  zuPumpID = i + 1;
                    //启动泵
                    start_pump(zuPumpID, pumpspeed);//泵控制启动
                    Thread.Sleep(500);
                }
            }
            //状态检测
            while (true)
            {
                curlv1 = sub_dev_Vave.m_PodSTA[0].waterLeve;
                curlv5 = sub_dev_Vave.m_PodSTA[4].waterLeve;
                curlv9 = sub_dev_Vave.m_PodSTA[8].waterLeve;

                if ((curlv1 - targetlv > 0)&&(addwaterSTA[0] == 0))
                {
                    Console.WriteLine("ID= 1 target reached" + " " + curlv1.ToString() + " return!");
                    addwaterSTA[0] = 1;
                    stopPump(1);//泵控制
                    Thread.Sleep(500);
                    sub_dev_Vave.SetVaves_status(1, 0);
                    //关闭阀门
                }
                if ((curlv5 - targetlv > 0) && (addwaterSTA[1] == 0))
                {
                    Console.WriteLine("ID= 5 target reached" + " " + curlv5.ToString() + " return!");
                    addwaterSTA[1] = 1;
                    stopPump(2);//泵控制
                    Thread.Sleep(500);
                    sub_dev_Vave.SetVaves_status(9, 0);
                    //关闭阀门
                }
                if ((curlv9 - targetlv > 0)&& (addwaterSTA[2] == 0))
                {
                    Console.WriteLine("ID= 9 target reached" + " " + curlv9.ToString() + " return!");
                    addwaterSTA[2] = 1;
                    stopPump(3);//泵控制
                    Thread.Sleep(500);
                    sub_dev_Vave.SetVaves_status(17, 0);
                    //关闭阀门
                }
                if (addwaterSTA[1] + addwaterSTA[2] + addwaterSTA[0] >= 3)
                {
                    break;
                }
            }
        }

        public void runOutWaterAuto_duli(int ID, int pumpspeed, float targetLv)
        {
            if (ID <= 0 || ID > 12) { return; }
            duliQuitFlag = 0;
            int i = ID - 1;
            float currentlv = sub_dev_Vave.m_PodSTA[i].waterLeve;

            float start_lv = currentlv;
            Stopwatch stopwatch = new Stopwatch();

            if (currentlv - targetLv < 0)
            {
                Console.WriteLine("ID=" + ID.ToString() + " target reached" + " " + targetLv.ToString() + " return!");
                return;
            }
            //打开进水阀门
            int openvavID_out = 2 * ID;
            int zuPumpID = (int)(i / 4) + 1;
            Console.WriteLine("出水 ID=" + ID.ToString() + " 阀门=" + openvavID_out.ToString() + " 泵+阀=" + zuPumpID.ToString());

            sub_dev_Vave.SetVaves_status(openvavID_out, 1);
            //切换出水组电磁阀
            start_pump_out(zuPumpID, 0);
            Thread.Sleep(500);
            //启动出泵
            stopwatch.Start();
            start_pump(zuPumpID, pumpspeed);//泵控制

            while (currentlv - targetLv > 2)
            {
                currentlv = sub_dev_Vave.m_PodSTA[i].waterLeve;
                Thread.Sleep(100);//每隔100 检查一次
                if (duliQuitFlag == 1)
                {
                    Console.WriteLine("outwater forced quit!");
                    return;
                }
            }
            stopwatch.Stop();
            TimeSpan elapsedTime = stopwatch.Elapsed;
            Console.WriteLine($"ID={ID}, from lv={start_lv} to lv={targetLv} with pumpspeed{pumpspeed}, " +
                $"the time is{elapsedTime},out target reached, stop! ");
            //关闭进水泵
            stopPump(zuPumpID);//泵控制
            Thread.Sleep(1000);
            //阀门控制
            sub_dev_Vave.SetVaves_status(openvavID_out, 0);
            //关闭阀门
        }

        public void StopALL_VAV()
        {
            Console.WriteLine("stop ALL pumps");
            stopPump(1);//泵控制
            Thread.Sleep(500);
            stopPump(2);//泵控制
            Thread.Sleep(500);
            stopPump(3);//泵控制
            Thread.Sleep(500);
            Console.WriteLine("close ALL VAVS");
            for (int i = 1; i <= 24; i++)
            {
                sub_dev_Vave.SetVaves_status(i, 0);
            }
        }
        //组 1，2，3
        public int AddwaterQuitFLG = 0;
        public void run_inWater_Zu(int Zu_ID, int pumpspeed, float targetLv)
        {
            if (Zu_ID <= 0 || Zu_ID > 3) { return; }
            AddwaterQuitFLG = 0;

            int lowpumpspeed = Math.Min(pumpspeed, 500);
            int realpumpspdcmd = pumpspeed;

            int [] ID = new int[4];//通路号
            int[] idx = new int[4];//索引号
            int[] ID_status = new int[4];//4个子通路状态
            float[] currentlev = new float[4];
            float[] lev_err = new float[4];

            int reachnum = 0;//已满通路数字
            for(int j=0;j<4;j++)
            {
                ID[j] = (Zu_ID-1)*4+j+1;
                idx[j] = ID[j] - 1;
                currentlev[j] = sub_dev_Vave.m_PodSTA[idx[j]].waterLeve;
                lev_err[j] = currentlev[j] - targetLv;
                if(lev_err[j]>=0)
                {
                    ID_status[j] = 1;//当前通路已满
                    reachnum++;
                    Console.WriteLine("通路已满：" + ID[j].ToString() + " level=" + currentlev[j].ToString() + " target=" + targetLv.ToString());
           //         realpumpspdcmd = lowpumpspeed;
                }
                else
                {
                    ID_status[j] = 0;//当前通路未满
                }
            }
            if(reachnum==4)//全部4个已经到达目标
            {
                Console.WriteLine("Zu_ID=" + Zu_ID.ToString() + "ALL target reached" + " " + targetLv.ToString() + " return!");
                for(int j=0;j<4;)
                {
                    Console.WriteLine("ID=" + ID[j].ToString() + " lv=" + currentlev[j].ToString());
                }
                return;
            }
            if(reachnum>=1)
            {
                realpumpspdcmd = lowpumpspeed;//有一个通路已满，速度换为龟速
            }

            //切换进水组电磁阀
            start_pump_in(Zu_ID, 0);
            //打开需要加水的瓶子阀门
            for (int j = 0; j < 4; j++)
            {
               if (ID_status[j] == 0)//打开未满通路进水阀门
                {
                    int openvavID1 = 2 * idx[j] + 1;
                    sub_dev_Vave.SetVaves_status(openvavID1, 1);
                    Console.WriteLine("进水 ID=" + ID[j].ToString() + " 阀门=" + openvavID1.ToString() + "打开");
                }
            }
            Thread.Sleep(500);
            //启动泵
            start_pump(Zu_ID, realpumpspdcmd);//泵控制启动
       
            //循环检查，并关闭对应阀门
            while(true)
            {
              //  int reachnum_dy = 0;
                for (int j = 0; j < 4; j++)//对每一个通路检查
                {
                    currentlev[j] = sub_dev_Vave.m_PodSTA[idx[j]].waterLeve;
                    lev_err[j] = currentlev[j] - targetLv;
                    //只检查未满的通路
                    if (ID_status[j] == 0)
                    {
                        if (lev_err[j] >= -1)//当前通路已满液
                        {
                            reachnum++;
                            ID_status[j] = 1;
                            Console.WriteLine("通路已满：" + ID[j].ToString() + " level=" + currentlev[j].ToString() + " target=" + targetLv.ToString());
                            if (reachnum >= 4)//全部满了
                            {
                                stopPump(Zu_ID);//泵控制,停止
                                Thread.Sleep(500);
                            }
                            else if (reachnum == 1)//第一个满了时，切换泵速度
                             { 
                                if (realpumpspdcmd != lowpumpspeed)//有一个通路已满时，采用低速
                                {
                                    Console.WriteLine("using low pump speed 500");
                                    realpumpspdcmd = lowpumpspeed;
                                    stopPump(Zu_ID);//泵控制
                                    Thread.Sleep(500);
                                    start_pump(Zu_ID, realpumpspdcmd);//泵控制
                                }

                            }
                            int openvavID1 = 2 * idx[j] + 1;
                            sub_dev_Vave.SetVaves_status(openvavID1, 0);//关闭阀门
                        }
                    }             
                 }
                if(reachnum >= 4)
                {
                    Console.WriteLine("加液体完成，正常退出！");
                    break;
                }
                if(AddwaterQuitFLG ==1)
                {
                    Console.WriteLine("外部指令退出！关闭所有阀门和泵");
                    stopPump(Zu_ID);//泵控制
                    Thread.Sleep(500);
                    for (int j = 0; j < 4; j++)
                    {
                         int openvavID1 = 2 * idx[j] + 1;
                         sub_dev_Vave.SetVaves_status(openvavID1, 0);
                         Console.WriteLine("进水 ID=" + ID[j].ToString() + " 阀门=" + openvavID1.ToString() + "关闭");
                    }
                    break;
                }
            }
        }


        public void run_outWater_Zu(int Zu_ID, int pumpspeed, float targetLv)
        {
            if (Zu_ID <= 0 || Zu_ID > 3) { return; }
            AddwaterQuitFLG = 0;

            int lowpumpspeed = Math.Min(pumpspeed, 500);
            int realpumpspdcmd = pumpspeed;

            int[] ID = new int[4];//通路号
            int[] idx = new int[4];//索引号
            int[] ID_status = new int[4];//4个子通路状态
            float[] currentlev = new float[4];
            float[] lev_err = new float[4];

            int reachnum = 0;//已达到目标通路数字
            for (int j = 0; j < 4; j++)
            {
                ID[j] = (Zu_ID - 1) * 4 + j + 1;
                idx[j] = ID[j] - 1;
                currentlev[j] = sub_dev_Vave.m_PodSTA[idx[j]].waterLeve;
                lev_err[j] = targetLv -currentlev[j];
                if (lev_err[j] >= 0)
                {
                    ID_status[j] = 1;//当前通路已达到目标
                    reachnum++;
                    Console.WriteLine("通路已达到目标：" + ID[j].ToString() + " level=" + currentlev[j].ToString() + " target=" + targetLv.ToString());
                    //         realpumpspdcmd = lowpumpspeed;
                }
                else
                {
                    ID_status[j] = 0;//当前通路未达到目标
                }
            }
            if (reachnum == 4)//全部4个已经到达目标
            {
                Console.WriteLine("Zu_ID=" + Zu_ID.ToString() + "ALL target reached" + " " + targetLv.ToString() + " return!");
                for (int j = 0; j < 4;)
                {
                    Console.WriteLine("ID=" + ID[j].ToString() + " lv=" + currentlev[j].ToString());
                }
                return;
            }
            if (reachnum >= 1)
            {
                realpumpspdcmd = lowpumpspeed;//有一个通路已满，速度换为龟速
            }

            //切换出水组电磁阀
            start_pump_out(Zu_ID, 0);
            //打开需要加水的瓶子阀门
            for (int j = 0; j < 4; j++)
            {
                if (ID_status[j] == 0)//打开未满通路进水阀门
                {
                    int openvavID1 = 2 * idx[j] + 2;
                    sub_dev_Vave.SetVaves_status(openvavID1, 1);
                    Console.WriteLine("出水 ID=" + ID[j].ToString() + " 阀门=" + openvavID1.ToString() + "打开");
                }
            }
            Thread.Sleep(500);
            //启动泵
            start_pump(Zu_ID, realpumpspdcmd);//泵控制启动

            //循环检查，并关闭对应阀门
            while (true)
            {
                //  int reachnum_dy = 0;
                for (int j = 0; j < 4; j++)//对每一个通路检查
                {
                    currentlev[j] = sub_dev_Vave.m_PodSTA[idx[j]].waterLeve;
                    lev_err[j] = targetLv - currentlev[j];
                    //只检查未满的通路
                    if (ID_status[j] == 0)
                    {
                        if (lev_err[j] >= 0)//当前通路已满液
                        {
                            reachnum++;
                            ID_status[j] = 1;
                            Console.WriteLine("通路已达到目标：" + ID[j].ToString() + " level=" + currentlev[j].ToString() + " target=" + targetLv.ToString());
                            if (reachnum >= 4)//全部满了
                            {
                                stopPump(Zu_ID);//泵控制,停止
                                Thread.Sleep(500);
                            }
                            else if (reachnum == 1)//第一个满了时，切换泵速度
                            {
                                if (realpumpspdcmd != lowpumpspeed)//有一个通路已满时，采用低速
                                {
                                    Console.WriteLine("using low pump speed 500");
                                    realpumpspdcmd = lowpumpspeed;
                                    stopPump(Zu_ID);//泵控制
                                    Thread.Sleep(500);
                                    start_pump(Zu_ID, realpumpspdcmd);//泵控制
                                }

                            }
                            int openvavID1 = 2 * idx[j] + 2;
                            sub_dev_Vave.SetVaves_status(openvavID1, 0);//关闭阀门
                        }
                    }
                }
                if (reachnum >= 4)
                {
                    Console.WriteLine("加液体完成，正常退出！");
                    break;
                }
                if (AddwaterQuitFLG == 1)
                {
                    Console.WriteLine("外部指令退出！关闭所有阀门和泵");
                    stopPump(Zu_ID);//泵控制
                    Thread.Sleep(500);
                    for (int j = 0; j < 4; j++)
                    {
                        int openvavID1 = 2 * idx[j] + 2;
                        sub_dev_Vave.SetVaves_status(openvavID1, 0);
                        Console.WriteLine("出水 ID=" + ID[j].ToString() + " 阀门=" + openvavID1.ToString() + "关闭");
                    }
                    break;
                }
            }
        }


















    }
}
