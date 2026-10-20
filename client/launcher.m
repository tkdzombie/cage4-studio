#import <Cocoa/Cocoa.h>

@interface Cage4Delegate : NSObject <NSApplicationDelegate>
@end

@implementation Cage4Delegate
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSString *script = [[NSBundle mainBundle] pathForResource:@"connect" ofType:@"sh"];
            NSString *directory = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Logs"];
            [[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:nil];
            NSString *path = [directory stringByAppendingPathComponent:@"CAGE4-Studio-Client.log"];
            if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
                [[NSFileManager defaultManager] createFileAtPath:path contents:nil attributes:nil];
            }
            NSFileHandle *log = [NSFileHandle fileHandleForWritingAtPath:path];
            [log seekToEndOfFile];
            [log writeData:[[NSString stringWithFormat:@"\n%@ Client starting\n", [NSDate date]] dataUsingEncoding:NSUTF8StringEncoding]];
            NSTask *task = [[NSTask alloc] init];
            task.executableURL = [NSURL fileURLWithPath:@"/bin/bash"];
            task.arguments = @[script ?: @"", @"start"];
            task.standardOutput = log;
            task.standardError = log;
            NSError *error = nil;
            BOOL launched = script && [task launchAndReturnError:&error];
            int status = 1;
            if (launched) {
                [task waitUntilExit];
                status = task.terminationStatus;
            } else {
                [log writeData:[[NSString stringWithFormat:@"Launch error: %@\n", error ?: @"connect.sh missing"] dataUsingEncoding:NSUTF8StringEncoding]];
            }
            [log writeData:[[NSString stringWithFormat:@"Client finished: %d\n", status] dataUsingEncoding:NSUTF8StringEncoding]];
            [log closeFile];
            dispatch_async(dispatch_get_main_queue(), ^{
                if (status != 0) {
                    NSAlert *alert = [[NSAlert alloc] init];
                    alert.messageText = @"CAGE4 连接失败";
                    alert.informativeText = @"请检查网络、SSH登录和服务器网页服务。详细日志：~/Library/Logs/CAGE4-Studio-Client.log";
                    [alert addButtonWithTitle:@"好的"];
                    [alert runModal];
                }
                [NSApp terminate:nil];
            });
        }
    });
}
@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyAccessory];
        __attribute__((objc_precise_lifetime)) Cage4Delegate *delegate = [[Cage4Delegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
