#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <pthread.h>
#include "Python.h"

#define   COUNT  3

/*摄像头参数id列表*/
#define BRIGHTNESS_ID 0x00980900
#define CONTRAST_ID 0x00980901
#define SATURATION_ID 0x00980902
#define HUE_ID 0x00980903
#define WHITE_BALANCE_TEMP_AUTO_ID 0x0098090c
#define GAMMA_ID 0x00980910
#define POWER_LINE_FREQUENCY_ID 0x00980918
#define WHITE_BALANCE_TEMP_ID 0x0098091a
#define SHARPNESS_ID 0x0098091b
#define BACKLIGHT_COMPENSATION_ID 0x0098091c
#define EXPOSURE_AUTO_ID 0x009a0901
#define EXPOSURE_ABSOLUTE_ID 0x009a0902
#define EXPOSURE_AUTO_PRIORITY_ID 0x009a0903

int video_fd ;
int cap_w;
int cap_h;
int length ;
unsigned char *yuv[COUNT] ;
unsigned int yuv_buf_len[COUNT];
struct v4l2_buffer  enqueue  , dequeue ;  //定义出入队的操作结构体成员
static pthread_t get_stream_pid;
unsigned char *save_buf = NULL;
pthread_mutex_t g_stream_mutex = PTHREAD_MUTEX_INITIALIZER;

int Init_Cameral(int Width , int Hight)
{
    printf("[%s][%d] Width=%d  Hight=%d\n", __func__, __LINE__, Width, Hight);
    //参数检查
    char *videodevname = NULL ;
    videodevname = "/dev/video0" ;

    //打开设备
    video_fd = open(videodevname , O_RDWR);
    if(-1 == video_fd )
    {
        perror("open video device fail");
        return -1 ;
    }

    cap_w = Width;
    cap_h = Hight;

    int i ;
    int ret ;

    struct v4l2_fmtdesc fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.index = 0;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    while ((ret = ioctl(video_fd, VIDIOC_ENUM_FMT, &fmt)) == 0)
    {
        fmt.index++;
        printf("pixelformat: %c%c%c%c, description=%s\n", fmt.pixelformat & 0xff,
                (fmt.pixelformat >> 8) & 0xff,
                (fmt.pixelformat >> 16) & 0xff,
                (fmt.pixelformat >> 24) & 0xff,
                fmt.description);
    }

    struct v4l2_format  format ;
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
    format.fmt.pix.width  = Width;
    format.fmt.pix.height = Hight;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV ;  //我支持的格式是这个

    ret = ioctl(video_fd , VIDIOC_S_FMT , &format);
    if(ret != 0)
    {
        perror("set video format fail");
        return -2 ;
    }

    //申请buffer,切割成几个部分
    //3
    struct v4l2_requestbuffers  requestbuffer ;
    requestbuffer.count = COUNT ;
    requestbuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
    requestbuffer.memory = V4L2_MEMORY_MMAP ;

    ret = ioctl(video_fd , VIDIOC_REQBUFS , &requestbuffer);
    if(ret != 0)
    {
        perror("request buffer fail ");
        return -3  ;
    }

    //querybuffer
    struct v4l2_buffer querybuffer ;
    querybuffer.type =  V4L2_BUF_TYPE_VIDEO_CAPTURE ;
    querybuffer.memory = V4L2_MEMORY_MMAP ;
    unsigned int buf_len = 0;

    for(i = 0 ; i < COUNT ; i++)
    {
        querybuffer.index = i ;

        ret = ioctl(video_fd , VIDIOC_QUERYBUF , &querybuffer);
        if(ret != 0)
        {
            perror("query buffer fail");
            return -4 ;
        }

        printf("index:%d length:%d  offset:%d \n" ,
                querybuffer.index , querybuffer.length , querybuffer.m.offset);
        length = querybuffer.length ;

        //将摄像头内存印射到进程的内存地址
        yuv[i] = mmap(0,querybuffer.length , PROT_READ | PROT_WRITE , MAP_SHARED , video_fd , querybuffer.m.offset );
        yuv_buf_len[i] = querybuffer.length;
        if (querybuffer.length > buf_len)
        {
            buf_len = querybuffer.length;
        }

        //列队
        struct v4l2_buffer  queuebuffer ;
        queuebuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
        queuebuffer.memory =  V4L2_MEMORY_MMAP ;
        queuebuffer.index = i ;

        ret = ioctl(video_fd , VIDIOC_QBUF , &queuebuffer);
        if(ret != 0)
        {
            perror("queuebuffer fail");
            return -5 ;
        }
    }

    save_buf = (unsigned char *)malloc(buf_len);
    if (NULL == save_buf)
    {
        printf("alloc buf len=%d failed:%m\n", buf_len);
        exit(-1);
    }

    //初始化入队出队
    enqueue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
    dequeue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
    enqueue.memory = V4L2_MEMORY_MMAP ;
    dequeue.memory = V4L2_MEMORY_MMAP ;

    return 0 ;
}

int Exit_Cameral(void)
{
    int i ;
    for(i = 0 ; i < COUNT ; i++)
        munmap(yuv+i , length);
    close(video_fd);
    usleep(300);

    printf("[%s][%d] success!\n", __func__, __LINE__);

    return 0 ;
}

void * get_stream_thread(void * para)
{
    int ret = 0;
    fd_set rdset;
    struct timeval tv;

    while (1)
    {
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        FD_ZERO(&rdset);
        FD_SET(video_fd, &rdset);

        ret = select(video_fd + 1, &rdset, NULL, NULL, &tv);
        if (-1 == ret)
        {
            printf("[%s][%d] select error:%m\n", __func__, __LINE__);
            usleep(300);
            continue;
        }
        else if (0 == ret)
        {
            printf("[%s][%d] select time out!\n", __func__, __LINE__);
            continue;
        }

        //出队
        ret = ioctl(video_fd , VIDIOC_DQBUF , &dequeue);
        if (ret != 0)
        {
            perror("dequeue fail");
            return NULL;
        }

        //printf("[%s][%d] dequeue.index=%d\n", __func__, __LINE__, dequeue.index);
        pthread_mutex_lock(&g_stream_mutex);
        memcpy(save_buf, yuv[dequeue.index], yuv_buf_len[dequeue.index]);
        pthread_mutex_unlock(&g_stream_mutex);

        enqueue.index = dequeue.index ;
        ret = ioctl(video_fd , VIDIOC_QBUF , &enqueue);
        if(ret != 0)
        {
            perror("enqueue fail");
            return NULL;
        }
    }

    return NULL;
}

int Start_Cameral(void)
{
    //开启摄像头
    int ret ;
    int on = 1 ;
    ret = ioctl(video_fd , VIDIOC_STREAMON , &on);
    if(ret != 0)
    {
        perror("start Cameral fail");
        return -1 ;
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&get_stream_pid, &attr, get_stream_thread, NULL);
    pthread_attr_destroy(&attr);

    usleep(1000);

    return 0 ;
}

int Stop_Cameral(void)
{
    //停止摄像头
    int ret ;
    int off= 1 ;
    ret = ioctl(video_fd , VIDIOC_STREAMOFF, &off);
    if(ret != 0)
    {
        perror("stop Cameral fail");
        return -1 ;
    }
    usleep(300);

    printf("[%s][%d] success!\n", __func__, __LINE__);

    return 0 ;
}

int convert_yuv_to_rgb_pixel(int y, int u, int v)
{
    unsigned int pixel32 = 0;
    unsigned char *pixel = (unsigned char *)&pixel32;
    int r, g, b;
    r = y + (1.370705 * (v-128));
    g = y - (0.698001 * (v-128)) - (0.337633 * (u-128));
    b = y + (1.732446 * (u-128));
    if(r > 255) r = 255;
    if(g > 255) g = 255;
    if(b > 255) b = 255;
    if(r < 0) r = 0;
    if(g < 0) g = 0;
    if(b < 0) b = 0;
    pixel[0] = r ;
    pixel[1] = g ;
    pixel[2] = b ;
    return pixel32;
}

int convert_yuv_to_rgb_buffer(unsigned char *yuv, unsigned char *rgb, unsigned int width, unsigned int height)
{
    unsigned int in, out = 0;
    unsigned int pixel_16;
    unsigned char pixel_24[3];
    unsigned int pixel32;
    int y0, u, y1, v;

    for(in = 0; in < width * height * 2; in += 4)
    {
        pixel_16 =
                        yuv[in + 3] << 24 |
                        yuv[in + 2] << 16 |
                        yuv[in + 1] <<  8 |
                        yuv[in + 0];
        y0 = (pixel_16 & 0x000000ff);
        u  = (pixel_16 & 0x0000ff00) >>  8;
        y1 = (pixel_16 & 0x00ff0000) >> 16;
        v  = (pixel_16 & 0xff000000) >> 24;
        pixel32 = convert_yuv_to_rgb_pixel(y0, u, v);
        pixel_24[0] = (pixel32 & 0x000000ff);
        pixel_24[1] = (pixel32 & 0x0000ff00) >> 8;
        pixel_24[2] = (pixel32 & 0x00ff0000) >> 16;
        rgb[out++] = pixel_24[0];
        rgb[out++] = pixel_24[1];
        rgb[out++] = pixel_24[2];
        pixel32 = convert_yuv_to_rgb_pixel(y1, u, v);
        pixel_24[0] = (pixel32 & 0x000000ff);
        pixel_24[1] = (pixel32 & 0x0000ff00) >> 8;
        pixel_24[2] = (pixel32 & 0x00ff0000) >> 16;
        rgb[out++] = pixel_24[0];
        rgb[out++] = pixel_24[1];
        rgb[out++] = pixel_24[2];
    }
    return 0;
}

int Get_Picture(unsigned char *buffer, int width, int height)
{
    pthread_mutex_lock(&g_stream_mutex);
    convert_yuv_to_rgb_buffer(save_buf, buffer, width, height);
    pthread_mutex_unlock(&g_stream_mutex);

    return 0 ;
}

int set_v4l2_para(int brightness, int contrast, int saturation, int sharpness)
{
#if 1
    struct v4l2_control ctrl;

    ctrl.id=BRIGHTNESS_ID;
    ctrl.value=brightness;
    if(ioctl(video_fd,VIDIOC_S_CTRL,&ctrl)==-1)
    {
        printf("set brightness=%d failed:%m\n", brightness);
        return -1;
    }

    ctrl.id=CONTRAST_ID;
    ctrl.value=contrast;
    if(ioctl(video_fd,VIDIOC_S_CTRL,&ctrl)==-1)
    {
        printf("set contrast=%d failed:%m\n", contrast);
        return -1;
    }

    ctrl.id=SATURATION_ID;
    ctrl.value=saturation;
    if(ioctl(video_fd,VIDIOC_S_CTRL,&ctrl)==-1)
    {
        printf("set saturation=%d failed:%m\n", saturation);
        return -1;
    }

    ctrl.id=SHARPNESS_ID;
    ctrl.value=sharpness;
    if(ioctl(video_fd,VIDIOC_S_CTRL,&ctrl)==-1)
    {
        printf("set sharpness=%d failed:%m\n", sharpness);
        return -1;
    }
    usleep(200000);
    //printf("set brightness=%d  contrast=%d  saturation=%d  sharpness=%d  success!\n", brightness, contrast, saturation, sharpness);
#endif

    return 0;
}