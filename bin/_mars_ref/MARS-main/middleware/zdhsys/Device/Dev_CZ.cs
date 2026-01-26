using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using System.Security.Cryptography;

namespace zdhsys
{
    public class Dev_CZ
    {
        public Dev_CZ() { 
            start_com();
            currentdata_cz = new chengzhongdata();
            rstPod = -1;
            rstPodval = -1;
        }

        DateTimeOffset nowOffset;
        DateTimeOffset nowOffset_lastmsg;// 用于计算相邻间隔时间

        Thread recvudp_chengzhong = null;//开启和接收udp线程
        //string data
        string udprcvdata_chengzhong;//
        UdpClient client_cz;//udp通信本机端
        IPAddress remoteIp_CZ;//设备地址
        IPEndPoint remoteEndPoint_CZ;//设备IP udp接收
        IPEndPoint remoteEndPoint_CZ_forsend;//设备IP udp发送
        int port = 5555;
        public int latest_dev_state;//最新设备状态
        public int latest_pod;//最新设备状态 pod
        public double latest_val;//最新设备反馈值
        public int timegap;


        public int rstPod;//接受到的称量的瓶子
        public double rstPodval;//称量结果
        bool newrstarrvied = false;

       public struct memrst//pod = 0; 表示大瓶子；pod = 1-12 小瓶子
        {
            public int pod;
            public double val;
        }
        public memrst latest_mem_rst;
        
        public memrst GetMemrst()
        {
            if(newrstarrvied)
            {
                newrstarrvied= false;
                latest_mem_rst.pod = rstPod;
                latest_mem_rst.val = rstPodval;  
            }
            else
            {
                Console.WriteLine("warning, no new mem result got!");
                latest_mem_rst.pod = -1;
                latest_mem_rst.val = -1;
            }
            return latest_mem_rst;
        }


        struct chengzhongdata
        {
            public int status;
            public int podID;
            public int val;
        }
        chengzhongdata currentdata_cz;

        struct chengzhongCMD
        {
            public int cmd;
            public int podID;
            public int addval;
        }
        chengzhongCMD currentCMD_cz;


        private void start_com()//初始化和启动udp
        {
            recvudp_chengzhong = new Thread(runrecv_chengzhong);
            recvudp_chengzhong.Start();
        }

        private bool checkconnection()//检查链接，10秒超时
        {
            nowOffset = DateTimeOffset.Now;
            int timestampOffset = (int)(nowOffset - nowOffset_lastmsg).TotalSeconds;
            timegap = timestampOffset;
            if (timestampOffset > 10)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        private void waitforok()//等待设备就绪，cmd == 8
        {
            while (true)
            {
                Thread.Sleep(500);
                if(checkconnection())
                {
                    if(latest_dev_state ==8)
                    {
                        break;
                    }
                    Console.WriteLine("not ready");
                    Console.WriteLine(latest_dev_state.ToString());
                }
              
            }
            Console.WriteLine("device ready 8");
        }
        //发送指令
        //cmd = 1 初始化
        //cmd = 3 通知设备已收到数据
        //cmd = 4 发送称大瓶子指令
        //cmd = 5 定量加液体指令 podID  val
        //cmd = 6 发送称小瓶子指令 podID
        private void sendcmd(int cmd, int ID, double val)//发送指令
        {
            //  waitforok();
            string cmdsend = "{\"cmd\": ,\"ID\":-1,\"val\":-1}";
            JObject objs = JObject.Parse(cmdsend);
            objs["cmd"] = cmd;
            objs["ID"] = ID;
            objs["val"] = val;

            string jsonstring = objs.ToString();

            Console.WriteLine(jsonstring);
            byte[] senddata = System.Text.Encoding.UTF8.GetBytes(jsonstring);
            Console.WriteLine(remoteEndPoint_CZ_forsend.ToString());
            client_cz.Send(senddata, senddata.Length, remoteEndPoint_CZ_forsend);
        }

        private void cmd_result()
        {
            while(true)
            {
                sendcmd(3, -1, -1);
                Thread.Sleep(500);

            }
            
        }
        //返回-1 表示链接断开或设备错误，其他，返回当前工作状态 8 空闲，9 错误，其他时候对应指令
        public int get_current_state()
        {
            if(checkconnection())
            {
                return latest_dev_state;
            }
            else
            {
                return -1;
            }
        }
        //等待设备ok，发送初始化，确认初始化指令已收到
        public bool sendcmd_initial_until_start()
        {
            waitforok();
            Console.WriteLine("check dev state ok, send cmd");
            int cnt = 0;
            while(true)
            {
                cnt++;
                sendcmd(1, -1, -1);//发送初始化命令
                Thread.Sleep(1000);
                int state = get_current_state();
                Console.WriteLine(state.ToString());
                if (state == 1)
                {
                    Console.WriteLine("send ok, device start initial");
                    return true;
                }
                if(cnt > 10)
                {
                    Console.WriteLine("send failed! ,time out break!");
                    return false;
                }
            }
        }
        //等待设备ok，发送指令，确认指令已收到
        public bool sendcmd_mem1_until_start()//称大瓶子
        {
            waitforok();
            Console.WriteLine("check dev state ok, send cmd");
            int cnt = 0;
            while (true)
            {
                cnt++;
                sendcmd(4, -1, -1);//发送命令
                Thread.Sleep(1000);
                int state = get_current_state();
                Console.WriteLine(state.ToString());
                if (state == 4)
                {
                    Console.WriteLine("send ok, device start initial");
                    return true;
                }
                if (cnt > 10)
                {
                    Console.WriteLine("send failed! ,time out break!");
                    return false;
                }
            }
        }
        //等待设备ok，发送指令，确认指令已收到
        public bool sendcmd_Add_until_start(int podID, double val)//加磁核
        {
            waitforok();
            Console.WriteLine("check dev state ok, send cmd");
            int cnt = 0;
            while (true)
            {
                cnt++;
                sendcmd(5, podID, val);//发送命令
                Thread.Sleep(1000);
                int state = get_current_state();
                Console.WriteLine(state.ToString());
                if (state == 5)
                {
                    Console.WriteLine("send ok, device start");
                    return true;
                }
                if (cnt > 10)
                {
                    Console.WriteLine("send failed! ,time out break!");
                    return false;
                }
            }
        }
        //等待设备ok，发送指令，确认指令已收到
        public bool sendcmd_Mem2_until_start(int podID)//称小瓶子
        {
            waitforok();
            Console.WriteLine("check dev state ok, send cmd");
            int cnt = 0;
            while (true)
            {
                cnt++;
                sendcmd(6, podID, -1);//发送命令
                Thread.Sleep(1000);
                int state = get_current_state();
                Console.WriteLine(state.ToString());
                if (state == 6)
                {
                    Console.WriteLine("send ok, device start");
                    return true;
                }
                if (cnt > 10)
                {
                    Console.WriteLine("send failed! ,time out break!");
                    return false;
                }
            }
        }



        //向设备反馈
        public bool sendcmd_havereceived()
        {
            sendcmd(3, -1, -1);//发送已收到指令
            rstPod = latest_pod;
            rstPodval = latest_val;
            return true;
        }



        //udp 初始化， 接受udp信息
        private void runrecv_chengzhong()
        {
            client_cz = new UdpClient(5555);
            remoteIp_CZ = IPAddress.Parse("192.168.1.222");
            int remotePort = 5555;
            remoteEndPoint_CZ = new IPEndPoint(remoteIp_CZ, remotePort);
            remoteEndPoint_CZ_forsend = new IPEndPoint(remoteIp_CZ, remotePort);
            CancellationTokenSource cts = new CancellationTokenSource();

            while (true)
            {
                try
                {
                    byte[] data = client_cz.Receive(ref remoteEndPoint_CZ);
                    if (data != null)
                    {
                        StringBuilder sb = new StringBuilder("");
                        sb.Append(Encoding.UTF8.GetString(data));
                      //  Console.WriteLine(sb);
                        udprcvdata_chengzhong = sb.ToString();
                      //  Console.WriteLine(udprcvdata_chengzhong);
                      try
                        {
                            string json_str_form = "{\"status\": 1,\"id\":0,\"val\":0}";
                            JToken jTokenform = JToken.Parse(json_str_form);
                            JObject objs = JObject.Parse(udprcvdata_chengzhong);

                            if (jTokenform.Type != objs.Type)
                            {
                                Console.WriteLine("JSON格式不正确！");
                            }
                            else
                            {
                               // Console.WriteLine("JSON格式正确！");
                                currentdata_cz.status = int.Parse(objs["status"].ToString());
                                currentdata_cz.podID = int.Parse(objs["id"].ToString());
                                currentdata_cz.val = int.Parse(objs["val"].ToString());
                                //objs[]
                                latest_dev_state = currentdata_cz.status;
                                if (currentdata_cz.status == 3)
                                {
                                    Console.WriteLine("got mesure val!");
                                    Console.WriteLine(currentdata_cz.podID.ToString());
                                    Console.WriteLine(currentdata_cz.val.ToString());
                                    latest_pod = currentdata_cz.podID;
                                    latest_val = currentdata_cz.val;
                                    sendcmd_havereceived();
                                    
                                }
                                nowOffset_lastmsg = DateTimeOffset.Now;

                            }
                        }
                        catch(Exception ex) {
                            Console.WriteLine(ex.Message.ToString());
                        }

                    }
                }
                catch (Exception ex)
                {
                    // tbudpdata.Text = ex.ToString();
                    //     Console.WriteLine(ex.ToString());
                    MessageBox.Show("UDP异常", ex.Message);
                    // break;
                }
                //   Thread.Sleep(1000);
            }

        }
        }


     
}
