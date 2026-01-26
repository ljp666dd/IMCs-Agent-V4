using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web;
using System.Windows.Forms;
using EasyModbus;
using System.IO.Pipes;
using System.Net.Sockets;
using Newtonsoft.Json;
using System.Diagnostics;
using System.Drawing;

namespace zdhsys
{
    internal partial class Dev_BG
    {

        public Dev_BG() 
        {
            sub_dev_Vave = new bg_in_out_water();
        }
      
        public bg_in_out_water sub_dev_Vave;

        public string chaoshengspd = "set_mixer_speed 15 ";//设置搅拌转速；
        public string chaoshengpower = "set_mixer_pow 15 ";//设置超声功率；

        public string mixstart = "mixer_start 15\r\n";
        public string mixend = "mixer_stop 15\r\n";

        public string tempset = "set_pid_temp 1 ";//设置温度
        public string tempstart = "pid_enable 1 1";//开始温控
        public string tempstop = "pid_enable 1 0";//开始温控

        public string ComPortName = "COM3";
        SerialPort serialPort_Dev_BG;

        public bool openPort()
        {
            serialPort_Dev_BG = new SerialPort(ComPortName, 115200, Parity.None, 8, StopBits.One);
            try
            {
                serialPort_Dev_BG.Open();
                //    throw new Exception();
                return true;
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message.ToString());
                return false;
            }

        }

        Thread lisn_th;
        public void start_lsn()
        {
            lisn_th = new Thread(lispro);
            lisn_th.Start();
        }

        public string rcvdata;
        bool checking = false;
        private void lispro()
        {  
            //byte[] receiveData = new byte[serialPort1.BytesToRead];//接收到的数据 serialPort1.Read(receiveData, 0, receiveData.Length);//读取数据
            while (serialPort_Dev_BG.IsOpen)
            {
                Thread.Sleep(100);
                if (checking)
                {
                    continue;
                }

                if (serialPort_Dev_BG.BytesToRead > 0)
                {
                    rcvdata = serialPort_Dev_BG.ReadLine();
                    Console.Write(rcvdata); // 输出到控制台
                                            //textBox1.Text = data;
                                            //   textBox1.Lines = data.Split('\n');
                }
                

            }

        }

        public class bg_dev_status_cfg
        {
            public int TotalSTA;//0 before initial; 1 initialed and idle; 2 working; 3 finish; 4 error
            public int [] pumpsta;//1号泵 对应1-4路
            public int [] pumpsta_curspd;
           // public int pump2sta;//2号泵 对应5-8路，
          //  public int pump3sta;//3号泵 对应9-12路

            public int [] pumpVavsta;//1号泵阀门，进1 出0
         //   public int pump2Vavsta;//2号泵阀门，进1 出0
         //   public int pump3Vavsta;//3号泵阀门，进1 出0

            public int motorUDsta;//电机上下状态，上0， 下1
            public int motorFBsta;//电机前后状态，后0，前1

            public int[] sonarSta;//12路超声状态
            public int[] sonarPowSet;//12路超声功率设置
            public int[] sonarSpeedSet;//12路超声速度设置

            public int[] heatSta;//12路加热状态
            public int[] heatTempSet;//12路加热温度设置

            public int[] waterLevLow;//清洗低位
            public int[] waterLevHigh;//清洗高位

            public int pumpSpd; //齿轮泵速度
            public int pumpSpdHigh;//齿轮泵高速
            public int pumpSpdLow;//齿轮泵低速度

            public int greepPumpSpd;//蠕动泵速度

            public bg_dev_status_cfg()
            {
                pumpsta = new int[3];
                pumpsta_curspd = new int[3];
                pumpVavsta = new int[3];

                sonarSta = new int[12];
                sonarPowSet = new int[12];
                heatSta = new int[12];
                heatTempSet = new int[12];
                waterLevLow = new int[12];
                waterLevHigh = new int[12];
                sonarSpeedSet = new int[12];

                pumpSpd = 500;//默认500

                for(int i=0;i<3;i++)
                {
                    pumpsta[i] = 0;
                    pumpsta_curspd[i] = 0;
                    pumpVavsta[i]= 0;
                }

                for(int i=0;i<12;i++)
                {
                    sonarSta[i] = 0;
                    sonarPowSet[i]= 20;
                    sonarSpeedSet[i] = 100;
                    heatSta[i] = 0;
                    heatTempSet[i] = 60;
                    waterLevHigh[i] = 55;
                }
            }

            public void loadfromcfg()
            {

            }
        }

        public bg_dev_status_cfg m_Status_cfg = new bg_dev_status_cfg();//当前设备状态与参数

        public void motorliftup()
        {
            string cmd1 = "motor_dir_set 1 0\r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);
            Thread.Sleep(500);

            string cmd2 = "motor1_2_run\r\n";
            byte[] array2 = System.Text.Encoding.ASCII.GetBytes(cmd2);
            serialPort_Dev_BG.Write(array2, 0, array2.Length);

            m_Status_cfg.motorUDsta = 0;
        }

        public void motorliftDowm()
        {
            string cmd1 = "motor_dir_set 1 1\r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);
            Thread.Sleep(500);

            string cmd2 = "motor1_2_run\r\n";
            byte[] array2 = System.Text.Encoding.ASCII.GetBytes(cmd2);
            serialPort_Dev_BG.Write(array2, 0, array2.Length);

            m_Status_cfg.motorUDsta = 1;
        }

        public void motorfront()
        {
            string cmd1 = "motor_dir_set 2 1\r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);
            Thread.Sleep(1000);

            string cmd2 = "motor3_4_run\r\n";
            byte[] array2 = System.Text.Encoding.ASCII.GetBytes(cmd2);
            serialPort_Dev_BG.Write(array2, 0, array2.Length);

            m_Status_cfg.motorFBsta = 1;
        }
        public void motorback()
        {
            string cmd1 = "motor_dir_set 2 0\r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);
            Thread.Sleep(1000);

            string cmd2 = "motor3_4_run\r\n";
            byte[] array2 = System.Text.Encoding.ASCII.GetBytes(cmd2);
            serialPort_Dev_BG.Write(array2, 0, array2.Length);

            m_Status_cfg.motorFBsta = 0;
        }

        //node : 1, 2
        //chn: 1-6,7
        //temp = 60
        public void set_temp(int node,int chn, int val)
        {
            string cmd = "set_pid_temp "+node.ToString()+" "+chn.ToString()+" "+val.ToString()+ "\r\n";
          //  string cmd = tempset + " " + val.ToString() + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            Thread.Sleep(500);
        }

        //node : 1, 2
        //chn: 1-6,7
        //action = 1
        public void start_temp(int node, int chn)
        {
            string cmd ="pid_enable " + node.ToString()+ " " +chn.ToString()+ " 1\r\n";
            // string cmd = tempstart + " " + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            Thread.Sleep(500);
        }
        //node : 1, 2
        //chn: 1-6
        //action = 0
        public void stop_temp(int node, int chn)
        {
            string cmd = "pid_enable " + node.ToString() + " " + chn.ToString() + " 0\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            Thread.Sleep(500);
        }

        //超声控制
        //测试代码,全开，15为全部发送
            public void set_sonar_rotationspd(int spd)
            {
                string cmd = chaoshengspd + " " + spd.ToString() + "\r\n";
                byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
                serialPort_Dev_BG.Write(array, 0, array.Length);
            }

            public void set_sonar_mixpower(int pow)
            {
                string cmd = chaoshengpower + " " + pow.ToString() + "\r\n";
                byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
                serialPort_Dev_BG.Write(array, 0, array.Length);
            }

            public void start_sonar()
            {
                string cmd0 = "set_mixer_pow 15 20\r\n";
                byte[] array0 = System.Text.Encoding.ASCII.GetBytes(cmd0);
                serialPort_Dev_BG.Write(array0, 0, array0.Length);
                Thread.Sleep(500);
                string cmd = mixstart + " 15 " + "\r\n";
                byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
                serialPort_Dev_BG.Write(array, 0, array.Length);
            }

            public void stop_sonar()
            {
                string cmd = mixend + " 15 " + "\r\n";
                byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
                serialPort_Dev_BG.Write(array, 0, array.Length);
            }

        //测试代码 独立开
        //说明：独立ID和实际ID关系待查
        //
        public void set_sonar_rotationspd(int ID, int spd)
        {
            string cmd = "set_mixer_speed "+ ID.ToString() + " " + spd.ToString() + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            Thread.Sleep(500);
        }

        public void set_sonar_mixpower(int ID, int pow)
        {
            string cmd = "set_mixer_pow " + ID.ToString() + " " + pow.ToString() + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            Thread.Sleep(500);
        }

        public void start_sonar(int ID)
        {
            string cmd = mixstart + " "+ID.ToString() + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
        }
        public void stop_sonar(int ID)
        {
            string cmd = mixend + " "+ID.ToString() + "\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
        }




        //超声控制



        public void runpump(int chn, int action, int speed)//启动停止泵，注意！泵2和3是反的，这里加了修正by jgl20240124
        {
            int realchn;
            if (chn == 2) { realchn = 3; } //2号和3号是反的
            else if (chn == 3) { realchn = 2; }//2号和3号是反的
            else if (chn == 1) { realchn = 1; }//1号是对的
            else { return; }//通道错误，不发数据
            string cmd = "run_gear_pump "+ realchn.ToString()+" " + action.ToString() + " " + speed.ToString() + " \r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);

            //状态记录采用原来的
            m_Status_cfg.pumpsta_curspd[chn - 1] = speed;
            m_Status_cfg.pumpsta[chn - 1] = action;
        }

        public void start_pump_out(int chn, int spd)//设置阀门为出水
        {
            string cmd = "run_valve_test "+chn.ToString()+" 0\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            //  runpump(1,2000);
            m_Status_cfg.pumpVavsta[chn - 1] = 0;
            Thread.Sleep(1000);
            
        }
        public void start_pump_in(int chn,int spd)//设置阀门为进水
        {
            string cmd = "run_valve_test " + chn.ToString() + " 1\r\n";
            byte[] array = System.Text.Encoding.ASCII.GetBytes(cmd);
            serialPort_Dev_BG.Write(array, 0, array.Length);
            m_Status_cfg.pumpVavsta[chn - 1] = 1;
            Thread.Sleep(1000);

        }

        public void start_pump(int chn,int spd)//启动泵，注意！泵2和3是反的，这里加了修正by jgl20240124
        {
         /*   int realchn;
            if(chn ==2){ realchn = 3; } //2号和3号是反的
            else if(chn==3){ realchn = 2; }//2号和3号是反的
            else if(chn==1){ realchn = 1; }//1号是对的
            else { return; }//通道错误，不发数据
            Thread.Sleep(1000);
            string cmd1 = "run_gear_pump " + realchn.ToString() + " 1 " + spd.ToString() + "\r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);*/
            runpump(chn,1, spd);
        }

        public ushort get_sta()
        {
            checking = true;
            Thread.Sleep(500);
            string cmd1 = "switchScan \r\n";
            byte[] array1 = System.Text.Encoding.ASCII.GetBytes(cmd1);
            serialPort_Dev_BG.Write(array1, 0, array1.Length);
            string data;
            ushort podsta = 0;

            int cnt = 0;
            int rcv = 0;

            while (true)
            {
                if (serialPort_Dev_BG.BytesToRead > 0)
                {
                   
                    data = serialPort_Dev_BG.ReadLine();

                    bool result = ushort.TryParse(data, out podsta); //i now = 108  
                    rcv++;
                   /// if(rcv == 2)
                   //{
                   if(result)
                    { 
                        podsta = ushort.Parse(data);
                        break;
                    }
                  
                //    Console.Write(rcvdata); // 输出到控制台
                }
                cnt++;
                if(cnt >30) { 
                    break; 
                }

                Thread.Sleep(100);
            }

            Thread.Sleep(500);
            checking = false;
            return podsta;
        }



        public void stopPump(int chn)
        {
            runpump(chn,0, 0);
        }

        public int cycles = 1;
        public int status = 0;
        Thread run_th;
        public bool isrunning;

        public void start_run()
        {
            isrunning = true;
            run_th = new Thread(runProcess);
            run_th.Start();
            //  run_th.Join();
        }

        public void stop_run()
        {
            if (run_th.IsAlive)
            {
                run_th.Abort();
            }
            Thread.Sleep(1000);
            StopALL_VAV();
            Thread.Sleep(1000);
            stop_sonar();
            status = 4;
            Thread.Sleep(1000);
            motorliftup();//X到位
            status = 6;
            Thread.Sleep(1000);
            motorback();
            Thread.Sleep(5 * 1000);
            status = 0;
            m_Status_cfg.TotalSTA = 1;

        }

        public void initialdev()
        {
            m_Status_cfg.TotalSTA = 2;
            StopALL_VAV();
            Thread.Sleep(1000);
            stop_sonar();
           // status = 4;
            Thread.Sleep(1000);
            motorliftup();//X到位
          //  status = 6;
            Thread.Sleep(1000);
            motorback();
            Thread.Sleep(10 * 1000);
         //   status = 0;
            m_Status_cfg.TotalSTA = 1;
        }


        public void runProcess()
        {
            status = 1;
            motorliftDowm();//到位
            Thread.Sleep(25 * 1000);        
            status = 2;
            for (int i = 0; i < cycles; i++)
            {
                //搅拌
                start_sonar();
                status = 3;
                Thread.Sleep(10 * 1000);
                //吸附
                stop_sonar();
                status = 4;
                motorfront();
                Thread.Sleep(45 * 1000);
                runOutWaterAuto_duli(1,500,30);
                Thread.Sleep(5 * 1000);
                runInWaterAuto_duli(1,500,50);
                motorback();
                Thread.Sleep(15 * 1000);
                start_sonar();
                Thread.Sleep(10 * 1000);
                stop_sonar();
                //   moveY_dir(true);//Y下降
                //     Thread.Sleep(10 * 1000);
                //  Thread.Sleep(10 * 1000);
                //进出水
                //    start_pump_in();
                //    Thread.Sleep(10 * 1000);
                // start_pump_out();//进水
                //  Thread.Sleep(10 * 1000);
                //   stopPump();
                //   moveY_dir(false);//Y下降
            }
            Thread.Sleep(2 * 1000);
            motorliftup();//X到位
            status = 6;
            Thread.Sleep(30 * 1000);
            status = 0;
            isrunning = false;

            m_Status_cfg.TotalSTA = 3;


        }

        public void  online_ctl_initial()
        {
            // stop_run();
            Thread th;
            th = new Thread(initialdev);
            th.Start();
        }

        public void online_ctl_run()
        {
            if (m_Status_cfg.TotalSTA != 1)
            {
                return;
            }
            m_Status_cfg.TotalSTA = 2;
            // stop_run();
           start_run();
        }



    }
}
