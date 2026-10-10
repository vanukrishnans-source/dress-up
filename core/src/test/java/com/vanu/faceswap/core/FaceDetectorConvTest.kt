package com.vanu.faceswap.core

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.GraphicsMode

/** Bitmap <-> RgbImage must keep R,G,B order (the AI models expect RGB input). */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class FaceDetectorConvTest {
    @Test fun rgbOrderAndRoundTrip() {
        val w = 5; val h = 3
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) for (x in 0 until w) bmp.setPixel(x, y, (0xff shl 24) or ((10 + x) shl 16) or ((100 + y) shl 8) or (200 + x + y))
        val img = FaceDetector.toRgb(bmp)
        assertEquals(w, img.width); assertEquals(h, img.height)
        for (y in 0 until h) for (x in 0 until w) {
            val i = (y * w + x) * 3
            assertEquals(10 + x, img.px[i].toInt() and 255)          // R
            assertEquals(100 + y, img.px[i + 1].toInt() and 255)     // G
            assertEquals(200 + x + y, img.px[i + 2].toInt() and 255) // B
        }
        val back = FaceDetector.toBitmap(img)
        for (y in 0 until h) for (x in 0 until w) assertEquals(bmp.getPixel(x, y), back.getPixel(x, y))
    }

    @Test fun rgb565InputIsConverted() {
        val bmp = Bitmap.createBitmap(4, 4, Bitmap.Config.RGB_565)
        bmp.eraseColor(0xFFFF0000.toInt())
        val img = FaceDetector.toRgb(bmp)
        assertEquals(255, img.px[0].toInt() and 255); assertEquals(0, img.px[1].toInt() and 255); assertEquals(0, img.px[2].toInt() and 255)
    }
}
